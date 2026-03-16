"""
Gymnasium Wrapper for Opentrons OT-2 Simulation
================================================

Wraps the pybullet OT-2 simulation as a Gymnasium environment for
reinforcement learning with Stable Baselines 3.

Observation space (6D):
    [pipette_x, pipette_y, pipette_z, target_x, target_y, target_z]

Action space (3D, continuous):
    [vx, vy, vz] — velocity commands for the 3 prismatic axes

Reward:
    - Negative Euclidean distance to target (dense reward)
    - Bonus +10 when error < 1mm
    - Small step penalty to encourage speed

Termination:
    - Success: error < 0.5mm for 10 consecutive steps
    - Timeout: max_steps reached (default 1000)

Author : Marin Chiosa
Course : BUas Applied AI & Data Science – Robotics (OT-2 Simulation)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import os


class OT2GymWrapper(gym.Env):
    """Gymnasium environment for OT-2 pipette positioning."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps=1000, num_substeps=10):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.num_substeps = num_substeps

        # Action space: velocity commands [vx, vy, vz] in [-1, 1]
        # Scaled to [-0.3, 0.3] m/s in step()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation space: [pipette_xyz, target_xyz]
        # Pipette can reach roughly [-0.2, 0.3] in X/Y, [0.1, 0.3] in Z
        # Target is sampled within the reachable workspace
        obs_low = np.array([-0.3, -0.3, 0.05, -0.3, -0.3, 0.05], dtype=np.float32)
        obs_high = np.array([0.4, 0.4, 0.35, 0.4, 0.4, 0.35], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Workspace bounds for random target sampling
        # Derived from URDF joint limits + pipette offset + robot base at [0,0,0.03]
        self.target_bounds = {
            'x': (-0.05, 0.15),
            'y': (-0.05, 0.15),
            'z': (0.15, 0.25),
        }

        # Velocity scaling
        self.action_scale = 0.3

        # Internal state
        self.sim = None
        self.robot_key = None
        self.target = None
        self.current_step = 0
        self.success_count = 0  # consecutive steps below threshold
        self.success_threshold = 0.0005  # 0.5mm
        self.success_steps_required = 10

    def _fix_baseplane(self):
        """Lock and hide the duplicate baseplane URDF."""
        base_id = self.sim.baseplaneId
        p.resetBasePositionAndOrientation(base_id, [0, 0, -10], [0, 0, 0, 1])
        for j in range(3):
            p.setJointMotorControl2(base_id, j, p.VELOCITY_CONTROL,
                                     targetVelocity=0, force=10000)
        for robot_id in self.sim.robotIds:
            p.setCollisionFilterPair(base_id, robot_id, -1, -1, enableCollision=0)
            for i in range(3):
                for j in range(3):
                    p.setCollisionFilterPair(base_id, robot_id, i, j, enableCollision=0)
                p.setCollisionFilterPair(base_id, robot_id, -1, j, enableCollision=0)
                p.setCollisionFilterPair(base_id, robot_id, i, -1, enableCollision=0)

    def _get_obs(self):
        """Read pipette position and combine with target."""
        state = self.sim.get_states()
        pos = np.array(state[self.robot_key]["pipette_position"], dtype=np.float32)
        obs = np.concatenate([pos, self.target.astype(np.float32)])
        return obs

    def _get_error(self, obs=None):
        """Euclidean distance from pipette to target."""
        if obs is None:
            obs = self._get_obs()
        return np.linalg.norm(obs[:3] - obs[3:6])

    def _sample_target(self):
        """Sample a random target within the reachable workspace."""
        tx = np.random.uniform(*self.target_bounds['x'])
        ty = np.random.uniform(*self.target_bounds['y'])
        tz = np.random.uniform(*self.target_bounds['z'])
        return np.array([tx, ty, tz], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment with a new random target."""
        super().reset(seed=seed)

        # Close existing sim if any
        if self.sim is not None:
            try:
                self.sim.close()
            except:
                pass

        # Create new simulation
        render = (self.render_mode == "human")
        self.sim = self._create_sim(render)
        self._fix_baseplane()

        # Get robot key
        state = self.sim.run([[0, 0, 0, 0]])
        self.robot_key = list(state.keys())[0]

        # Sample new target
        self.target = self._sample_target()
        self.current_step = 0
        self.success_count = 0

        obs = self._get_obs()
        info = {"error": self._get_error(obs), "target": self.target.copy()}

        return obs, info

    def _create_sim(self, render):
        """Create the simulation, handling texture directory requirements."""
        # Ensure texture directories exist
        os.makedirs('textures/_plates', exist_ok=True)
        if os.path.exists('uvmapped_dish_large_comp.png'):
            import shutil
            if not os.path.exists('textures/texture1.png'):
                shutil.copy('uvmapped_dish_large_comp.png', 'textures/texture1.png')
            if not os.path.exists('textures/_plates/plate1.png'):
                shutil.copy('uvmapped_dish_large_comp.png', 'textures/_plates/plate1.png')

        from sim_class import Simulation
        return Simulation(num_agents=1, render=render)

    def step(self, action):
        """Execute one PID-period step (multiple physics substeps)."""
        # Scale action from [-1, 1] to [-action_scale, action_scale]
        action = np.clip(action, -1.0, 1.0)
        vx = float(action[0]) * self.action_scale
        vy = float(action[1]) * self.action_scale
        vz = float(action[2]) * self.action_scale

        # Apply action for multiple physics substeps
        self.sim.run([[vx, vy, vz, 0]], num_steps=self.num_substeps)

        self.current_step += 1

        # Get observation and error
        obs = self._get_obs()
        error = self._get_error(obs)

        # ── Reward ───────────────────────────────────────────────
        # Dense: negative distance (closer = less negative = better)
        reward = -error * 100  # scale up so gradients are meaningful

        # Bonus for getting very close
        if error < 0.001:  # < 1mm
            reward += 10.0
        if error < 0.0005:  # < 0.5mm
            reward += 20.0

        # Small step penalty to encourage reaching target quickly
        reward -= 0.1

        # ── Termination ──────────────────────────────────────────
        # Success: stay below threshold for N consecutive steps
        if error < self.success_threshold:
            self.success_count += 1
        else:
            self.success_count = 0

        terminated = (self.success_count >= self.success_steps_required)
        truncated = (self.current_step >= self.max_steps)

        if terminated:
            reward += 100.0  # big bonus for successful completion

        info = {
            "error": error,
            "target": self.target.copy(),
            "step": self.current_step,
            "success_count": self.success_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Rendering is handled by pybullet GUI when render_mode='human'."""
        pass

    def close(self):
        """Clean up the simulation."""
        if self.sim is not None:
            try:
                self.sim.close()
            except:
                pass
            self.sim = None
