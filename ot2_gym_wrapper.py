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
    
    def __init__(self, render=False, max_steps=300, target_threshold=0.001):
        super(OT2Env, self).__init__()
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold  # 0.001m = 1mm

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
        
        # Previous distance (for reward shaping)
        self.previous_distance = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Set a random goal position within the working area
        # Workspace limits: X [0.0, 0.15], Y [-0.05, 0.15], Z [0.17, 0.25]
        self.goal_position = np.array([
            np.random.uniform(0.0, 0.15),    # X
            np.random.uniform(-0.05, 0.15),  # Y
            np.random.uniform(0.17, 0.25)    # Z
        ], dtype=np.float32)
        
        # Reset simulation
        state = self.sim.reset(num_agents=1)
        
        # Extract pipette position
        robot_key = list(state.keys())[0]
        pipette_pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        # Reset counters
        self.steps = 0
        self.previous_distance = np.linalg.norm(pipette_pos - self.goal_position)
        
        info = {}
        return observation, info

    def step(self, action):
        # Append 0 for drop action
        action = np.append(action, 0)

        # Execute action
        state = self.sim.run([action])

        # Extract pipette position
        robot_key = list(state.keys())[0]
        pipette_pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        observation = np.concatenate([pipette_pos, self.goal_position]).astype(np.float32)

        # Calculate distance to goal
        distance = np.linalg.norm(pipette_pos - self.goal_position)
        
        # =====================================================================
        # REWARD FUNCTION
        # =====================================================================
        
        # 1. Base reward: negative distance (scaled)
        reward = -distance * 10
        
        # 2. Progress reward: bonus for getting closer
        distance_improvement = self.previous_distance - distance
        reward += distance_improvement * 100
        
        # 3. Threshold bonuses
        if distance < 0.01:    # < 10mm
            reward += 1.0
        if distance < 0.005:   # < 5mm
            reward += 2.0
        if distance < self.target_threshold:  # < 1mm (target)
            reward += 5.0
        
        # 4. Small time penalty
        reward -= 0.01
        
        # Update previous distance
        self.previous_distance = distance

        # Check termination (goal reached)
        if distance < self.target_threshold:
            terminated = True
            reward += 100  # Big bonus for success!
        else:
            terminated = False
        
        # Convert to Python float
        reward = float(reward)

        # Check truncation (max steps)
        self.steps += 1
        truncated = self.steps >= self.max_steps

        info = {
            'distance': distance,
            'distance_mm': distance * 1000,
            'success': distance < self.target_threshold
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        try:
            self.sim.close()
        except:
            pass