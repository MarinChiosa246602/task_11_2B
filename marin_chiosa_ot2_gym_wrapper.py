import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=500, target_threshold=0.001):
        super(OT2Env, self).__init__()
        
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        
        self.sim = Simulation(num_agents=1, render=render)
        self._fix_baseplane()
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # OT-2 workspace bounds
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
        self.initial_distance = None
        self.prev_distance = None
        self.best_distance = None
    
    def _fix_baseplane(self):
        """Fix duplicate baseplane URDF interference."""
        try:
            base_id = self.sim.baseplaneId
            p.resetBasePositionAndOrientation(base_id, [0, 0, -10], [0, 0, 0, 1])
            num_joints = p.getNumJoints(base_id)
            for joint_idx in range(num_joints):
                p.setJointMotorControl2(
                    base_id, joint_idx, p.VELOCITY_CONTROL,
                    targetVelocity=0, force=10000
                )
            for robot_id in self.sim.robotIds:
                p.setCollisionFilterPair(base_id, robot_id, -1, -1, enableCollision=0)
                for i in range(num_joints):
                    p.setCollisionFilterPair(base_id, robot_id, i, -1, enableCollision=0)
                    for j in range(p.getNumJoints(robot_id)):
                        p.setCollisionFilterPair(base_id, robot_id, i, j, enableCollision=0)
        except Exception as e:
            print(f"Warning: _fix_baseplane failed: {e}")
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Random goal within workspace
        self.goal_position = np.random.uniform(
            self.workspace_low, self.workspace_high
        ).astype(np.float32)
        
        state_dict = self.sim.reset(num_agents=1)
        self._fix_baseplane()
        
        current_pos = self._extract_position(state_dict)
        
        self.initial_distance = float(np.linalg.norm(current_pos - self.goal_position))
        self.prev_distance = self.initial_distance
        self.best_distance = self.initial_distance
        self.steps = 0
        
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        return observation, {}
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        
        # Scale action — lower max velocity for finer control
        max_velocity = 1.5
        velocity = action * max_velocity
        
        full_action = [float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]
        
        # SINGLE sim step for maximum control resolution
        state_dict = self.sim.run([full_action], num_steps=1)
        
        current_pos = self._extract_position(state_dict)
        distance = float(np.linalg.norm(current_pos - self.goal_position))
        
        reward = self._calculate_reward(distance)
        
        terminated = bool(distance < self.target_threshold)
        self.steps += 1
        truncated = bool(self.steps >= self.max_steps)
        
        # Track for reward calculation
        self.prev_distance = distance
        if distance < self.best_distance:
            self.best_distance = distance
        
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        info = {
            'distance_to_goal': distance,
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist(),
            'best_distance': self.best_distance,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, distance):
        """
        Multi-component reward designed for sub-mm precision:
        
        1. Exponential closeness: -exp(k * distance) 
           → Reward grows EXPONENTIALLY as agent gets closer
           → Agent cares 10x more about 1mm→0mm than 10mm→9mm
        
        2. Progress reward: bonus for improving over previous step
           → Incentivizes continuous movement toward goal
        
        3. New-best bonus: extra reward for beating the episode best
           → Prevents settling at "good enough" distances
        
        4. Success bonus: large reward for reaching threshold
        
        5. Time penalty: small per-step cost to encourage efficiency
        """
        # 1. Exponential distance penalty (main signal)
        # At 10mm: -exp(50*0.01) = -1.65
        # At 1mm:  -exp(50*0.001) = -1.05
        # At 0.1mm: -exp(50*0.0001) = -1.005
        # The gradient is STEEPER when closer — agent learns to be precise
        exp_penalty = -np.exp(50.0 * distance) + 1.0  # Shift so 0 distance = 0 penalty
        
        # 2. Progress reward
        improvement = self.prev_distance - distance
        progress_reward = 20.0 * improvement  # Positive when getting closer
        
        # 3. New-best bonus
        new_best_bonus = 0.0
        if distance < self.best_distance:
            new_best_bonus = 5.0
        
        # 4. Success bonus (scaled by how fast)
        success_bonus = 0.0
        if distance < self.target_threshold:
            # Bigger bonus for reaching goal faster
            time_bonus = max(0, 1.0 - self.steps / self.max_steps)
            success_bonus = 100.0 + 50.0 * time_bonus
        
        # 5. Small time penalty
        time_penalty = -0.05
        
        reward = exp_penalty + progress_reward + new_best_bonus + success_bonus + time_penalty
        return float(reward)
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()
    
    def _extract_position(self, state_dict):
        robotId = list(sorted(state_dict.keys()))[0]
        robot_state = state_dict.get(robotId, {})
        return np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
    
    def _normalize_position(self, position):
        return (2.0 * (position - self.workspace_low) / 
                (self.workspace_high - self.workspace_low) - 1.0).astype(np.float32)