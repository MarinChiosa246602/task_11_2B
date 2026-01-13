"""
OT-2 Gymnasium Environment Wrapper
==================================
FIXED VERSION - Uses goal-relative observations so the model
learns to move TOWARDS the goal, not in a fixed direction.

Key insight: The observation tells the model "the goal is in THIS direction"
so the optimal policy is simply "move in that direction".
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Gymnasium environment for OT-2 pipette position control.
    
    Observation Space (10 dimensions):
        - Relative position to goal (dx, dy, dz) - NOT normalized, actual offset
        - Distance to goal (1 value, normalized)
        - Current pipette position (x, y, z)
        - Goal position (x, y, z)
    
    Action Space (3 dimensions):
        - Velocity commands (vx, vy, vz) in range [-1, 1]
    """
    
    def __init__(self, render=False, max_steps=100, target_threshold=0.001):
        super(OT2Env, self).__init__()
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold

        self.sim = Simulation(num_agents=1, render=render)

        # Action space: velocity commands [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [delta_x, delta_y, delta_z, distance_norm, pipette_xyz, goal_xyz]
        self.observation_space = spaces.Box(
            low=np.array([-0.5, -0.5, -0.5, 0.0, -0.5, -0.5, 0.0, -0.5, -0.5, 0.0], dtype=np.float32),
            high=np.array([0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            dtype=np.float32
        )

        self.steps = 0
        self.goal_position = None
        self.previous_distance = None

    def _get_observation(self, pipette_pos):
        """Create observation with goal-relative information."""
        # Delta to goal (this is the KEY - tells model where to go)
        delta = self.goal_position - pipette_pos
        distance = np.linalg.norm(delta)
        
        # Normalize distance (workspace is ~0.3m max)
        distance_norm = min(distance / 0.3, 1.0)
        
        # Full observation
        obs = np.concatenate([
            delta,                    # 3: direction/offset to goal (MOST IMPORTANT)
            [distance_norm],          # 1: normalized distance
            pipette_pos,              # 3: current position
            self.goal_position        # 3: goal position
        ]).astype(np.float32)
        
        return obs, distance

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Random goal within workspace
        self.goal_position = np.array([
            np.random.uniform(0.0, 0.15),
            np.random.uniform(-0.05, 0.15),
            np.random.uniform(0.17, 0.25)
        ], dtype=np.float32)
        
        state = self.sim.reset(num_agents=1)
        robot_key = list(state.keys())[0]
        pipette_pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        
        obs, distance = self._get_observation(pipette_pos)
        
        self.steps = 0
        self.previous_distance = distance
        
        return obs, {}

    def step(self, action):
        action_full = np.append(action, 0)
        state = self.sim.run([action_full])
        
        robot_key = list(state.keys())[0]
        pipette_pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        
        obs, distance = self._get_observation(pipette_pos)
        
        # =====================================================================
        # REWARD FUNCTION - Designed for proper learning
        # =====================================================================
        
        # 1. Progress reward (MOST IMPORTANT) - reward getting closer
        progress = self.previous_distance - distance
        reward = progress * 500  # Strong signal
        
        # 2. Direction alignment bonus
        # The delta (obs[:3]) tells model where to go
        # Reward when action points in same direction as delta
        delta = obs[:3]
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 0.001:
            direction = delta / delta_norm
            alignment = np.dot(action, direction)  # -1 to +1
            reward += alignment * 5.0  # Bonus for correct direction
        
        # 3. Distance-based shaping (small penalty for being far)
        reward -= distance * 10
        
        # 4. Success bonus
        terminated = False
        if distance < self.target_threshold:
            reward += 200  # Big bonus for reaching goal!
            terminated = True
        elif distance < 0.005:  # < 5mm
            reward += 20
        elif distance < 0.01:   # < 10mm
            reward += 5
        
        # 5. Small time penalty
        reward -= 0.5
        
        self.previous_distance = distance
        self.steps += 1
        
        truncated = self.steps >= self.max_steps
        
        info = {
            'distance': distance,
            'distance_mm': distance * 1000,
            'success': distance < self.target_threshold
        }

        return obs, float(reward), terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        try:
            self.sim.close()
        except:
            pass
