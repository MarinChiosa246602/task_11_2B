"""
OT-2 Gymnasium Environment Wrapper
==================================
A Gymnasium-compatible wrapper for the Opentrons OT-2 simulation.
Follows the provided template structure.

Author: Task 11 - RL Controller
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Gymnasium environment for OT-2 pipette position control.
    
    Observation Space (6 dimensions):
        - Current pipette position (x, y, z)
        - Goal position (x, y, z)
    
    Action Space (3 dimensions):
        - Velocity commands (vx, vy, vz) in range [-1, 1]
    """
    
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render_mode = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action space: velocity commands (vx, vy, vz) normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space: [pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(
            low=np.array([-0.5, -0.5, 0.0, -0.5, -0.5, 0.0], dtype=np.float32),
            high=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            dtype=np.float32
        )

        # Keep track of the number of steps
        self.steps = 0
        
        # Goal position (set in reset)
        self.goal_position = None

    def reset(self, seed=None, options=None):
        # Being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position within the working area
        # Workspace limits: X [0.0, 0.15], Y [-0.05, 0.15], Z [0.17, 0.25]
        self.goal_position = np.array([
            np.random.uniform(0.0, 0.15),    # X
            np.random.uniform(-0.05, 0.15),  # Y
            np.random.uniform(0.17, 0.25)    # Z (must be >= 0.17 to be reachable)
        ], dtype=np.float32)
        
        # Call the environment reset function
        state = self.sim.reset(num_agents=1)
        
        # Process the observation: extract pipette position, append goal position
        robot_key = list(state.keys())[0]
        pipette_pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        # Reset the number of steps
        self.steps = 0
        
        # Return observation and info dict (Gymnasium API)
        info = {}

        return observation, info

    def step(self, action):
        # Execute one time step within the environment
        # Since we are only controlling the pipette position, we accept 3 values 
        # for the action and need to append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        # We pass action as a list because sim.run expects a list of actions (one per agent)
        state = self.sim.run([action])

        # Process the observation: extract pipette position, append goal position
        robot_key = list(state.keys())[0]
        pipette_pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        # Calculate the distance to goal
        distance = np.linalg.norm(pipette_pos - self.goal_position)
        
        # Calculate the reward
        # Negative distance encourages getting closer to the goal
        reward = -distance

        # Check if the task has been completed (terminated)
        # Threshold of 0.001m (1mm) - reasonable for pipette tip precision
        if distance < 0.001:
            terminated = True
            reward += 100  # Positive reward for completing the task
        else:
            terminated = False
        
        # Convert reward to Python float (required by SB3)
        reward = float(reward)

        # Check if the episode should be truncated (max steps reached)
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {}  # We don't need to return any additional information

        # Increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        try:
            self.sim.close()
        except:
            pass  # Already closed