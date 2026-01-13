"""
OT-2 Gymnasium Environment Wrapper
==================================
FIXED: Observation gives NORMALIZED direction so optimal action = obs[:3]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Observation (4 dim): 
        - direction_x, direction_y, direction_z (NORMALIZED, range [-1, 1])
        - distance_normalized (range [0, 1])
    
    OPTIMAL POLICY: action = observation[:3]
    """
    
    def __init__(self, render=False, max_steps=100, target_threshold=0.001):
        super(OT2Env, self).__init__()
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold

        self.sim = Simulation(num_agents=1, render=render)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 4D observation: [dir_x, dir_y, dir_z, dist_norm]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.steps = 0
        self.goal_position = None
        self.previous_distance = None
        self.pipette_position = None

    def _get_observation(self, pipette_pos):
        """Get NORMALIZED direction to goal."""
        self.pipette_position = pipette_pos
        delta = self.goal_position - pipette_pos
        distance = np.linalg.norm(delta)
        
        # NORMALIZED direction (this is what the action should match!)
        if distance > 0.0001:
            direction = delta / distance  # Unit vector, range [-1, 1]
        else:
            direction = np.zeros(3)
        
        # Normalized distance
        distance_norm = min(distance / 0.3, 1.0)
        
        obs = np.concatenate([direction, [distance_norm]]).astype(np.float32)
        return obs, distance

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

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
        # REWARD: Encourage action to match direction (obs[:3])
        # =====================================================================
        
        direction = obs[:3]  # Normalized direction to goal
        
        # 1. ALIGNMENT REWARD - action should equal direction
        # dot product: 1.0 if perfectly aligned, -1.0 if opposite
        alignment = np.dot(action, direction)
        reward = alignment * 100  # Strong reward for matching direction
        
        # 2. PROGRESS REWARD
        progress = self.previous_distance - distance
        reward += progress * 500
        
        # 3. SUCCESS BONUS
        terminated = False
        if distance < self.target_threshold:
            reward += 1000
            terminated = True
        elif distance < 0.005:
            reward += 50
        elif distance < 0.01:
            reward += 10
        
        # 4. Time penalty
        reward -= 1.0
        
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
