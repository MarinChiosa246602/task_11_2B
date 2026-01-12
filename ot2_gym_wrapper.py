"""
OT-2 Gymnasium Environment Wrapper
==================================
IMPROVED VERSION with better reward shaping that actually trains!

Key improvements:
1. Normalized observations for better learning
2. Stronger progress-based rewards
3. Better termination conditions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Gymnasium environment for OT-2 pipette position control.
    """
    
    def __init__(self, render=False, max_steps=300, target_threshold=0.001):
        super(OT2Env, self).__init__()
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold  # 1mm

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action space: velocity commands normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(
            low=np.array([-0.5, -0.5, 0.0, -0.5, -0.5, 0.0], dtype=np.float32),
            high=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            dtype=np.float32
        )

        self.steps = 0
        self.goal_position = None
        self.previous_distance = None
        self.initial_distance = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Random goal within workspace
        self.goal_position = np.array([
            np.random.uniform(0.0, 0.15),
            np.random.uniform(-0.05, 0.15),
            np.random.uniform(0.17, 0.25)
        ], dtype=np.float32)
        
        # Reset simulation
        state = self.sim.reset(num_agents=1)
        
        robot_key = list(state.keys())[0]
        pipette_pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        self.steps = 0
        self.previous_distance = np.linalg.norm(pipette_pos - self.goal_position)
        self.initial_distance = self.previous_distance
        
        return observation, {}

    def step(self, action):
        # Append drop action (0)
        action_full = np.append(action, 0)

        # Execute action
        state = self.sim.run([action_full])

        robot_key = list(state.keys())[0]
        pipette_pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        # Calculate distance
        distance = np.linalg.norm(pipette_pos - self.goal_position)
        
        # =====================================================================
        # IMPROVED REWARD FUNCTION
        # =====================================================================
        
        # 1. Progress reward - reward for getting closer (MOST IMPORTANT)
        distance_improvement = self.previous_distance - distance
        reward = distance_improvement * 1000  # Strong signal for progress
        
        # 2. Distance-based penalty (normalized)
        normalized_distance = distance / (self.initial_distance + 1e-8)
        reward -= normalized_distance * 0.1
        
        # 3. Success bonus
        terminated = False
        if distance < self.target_threshold:
            reward += 100  # Big bonus for reaching goal!
            terminated = True
        elif distance < 0.005:  # < 5mm bonus
            reward += 5
        elif distance < 0.01:   # < 10mm bonus
            reward += 1
        
        # 4. Small time penalty
        reward -= 0.1
        
        # Update state
        self.previous_distance = distance
        self.steps += 1
        
        # Check truncation
        truncated = self.steps >= self.max_steps
        
        info = {
            'distance': distance,
            'distance_mm': distance * 1000,
            'success': distance < self.target_threshold
        }

        return observation, float(reward), terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        try:
            self.sim.close()
        except:
            pass