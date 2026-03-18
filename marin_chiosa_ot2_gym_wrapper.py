import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Custom Gymnasium environment for OT-2 robot control.
    
    Observation Space: 6D [current_x, current_y, current_z, goal_x, goal_y, goal_z] (normalized)
    Action Space: 3D [x, y, z] velocities normalized to [-1, 1]
    Reward: Time penalty + Distance penalty + Success bonus
    
    Parameters
    ----------
    render : bool
        Whether to render the simulation visually
    max_steps : int
        Maximum steps per episode before truncation (default: 300)
    target_threshold : float
        Distance threshold (meters) for successful goal achievement (default: 0.005 = 5mm)
    """
    
    def __init__(self, render=False, max_steps=300, target_threshold=0.005):
        super(OT2Env, self).__init__()
        
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        
        # Create simulation
        self.sim = Simulation(num_agents=1, render=render)
        
        # Fix baseplane: the sim loads the robot URDF twice — body 0 (scenery)
        # overlaps with body 1 (controllable robot). We move body 0 underground,
        # lock its joints, and disable collision to prevent interference.
        self._fix_baseplane()
        
        # Define action space: normalized [-1, 1] for RL algorithms
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space: 6D normalized positions
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # OT-2 workspace bounds (verified from simulation)
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
        self.initial_distance = None
    
    def _fix_baseplane(self):
        """Fix the duplicate baseplane URDF that interferes with the robot."""
        try:
            base_id = self.sim.baseplaneId
            
            # Move far underground
            p.resetBasePositionAndOrientation(base_id, [0, 0, -10], [0, 0, 0, 1])
            
            # Lock however many joints exist
            num_joints = p.getNumJoints(base_id)
            for joint_idx in range(num_joints):
                p.setJointMotorControl2(
                    base_id, joint_idx, p.VELOCITY_CONTROL,
                    targetVelocity=0, force=10000
                )
            
            # Disable collision with robot
            for robot_id in self.sim.robotIds:
                p.setCollisionFilterPair(base_id, robot_id, -1, -1, enableCollision=0)
                for i in range(num_joints):
                    p.setCollisionFilterPair(base_id, robot_id, i, -1, enableCollision=0)
                    for j in range(p.getNumJoints(robot_id)):
                        p.setCollisionFilterPair(base_id, robot_id, i, j, enableCollision=0)
        except Exception as e:
            print(f"Warning: _fix_baseplane failed: {e} (continuing without fix)")
    
    def reset(self, seed=None):
        """Reset environment to initial state with new random goal."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random goal within workspace
        self.goal_position = np.random.uniform(
            self.workspace_low,
            self.workspace_high
        ).astype(np.float32)
        
        # Reset simulation
        state_dict = self.sim.reset(num_agents=1)
        
        # Re-apply baseplane fix after reset
        self._fix_baseplane()
        
        # Extract current position
        current_pos = self._extract_position(state_dict)
        
        # Store initial distance for reward scaling
        self.initial_distance = float(np.linalg.norm(current_pos - self.goal_position))
        
        # Create normalized observation
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Reset step counter
        self.steps = 0
        
        return observation, {}
    
    def step(self, action):
        """Execute one step in the environment."""
        action = np.asarray(action, dtype=np.float32)
        
        # Scale action to velocity range
        max_velocity = 2.0
        velocity = action * max_velocity
        
        # Create full action array [vx, vy, vz, drop]
        full_action = [float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]

        # Execute action in simulation (multiple substeps for stability)
        state_dict = self.sim.run([full_action], num_steps=5)
        
        # Extract current position
        current_pos = self._extract_position(state_dict)
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
        
        # Calculate reward
        reward = self._calculate_reward(distance_to_goal)
        
        # Check if goal reached
        terminated = bool(distance_to_goal < self.target_threshold)
        
        # Increment step counter
        self.steps += 1
        
        # Check if max steps reached
        truncated = bool(self.steps >= self.max_steps)
        
        # Create observation
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Info for logging
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, distance_to_goal):
        """
        Reward function:
        1. Time penalty: -0.1 per step (punish slow movement)
        2. Distance penalty: -10 * distance (punish being far from goal)
        3. Success bonus: +50 (big reward for reaching goal)
        """
        time_penalty = -0.1
        distance_penalty = -10.0 * distance_to_goal
        success_bonus = 50.0 if distance_to_goal < self.target_threshold else 0.0
        
        reward = time_penalty + distance_penalty + success_bonus
        return float(reward)
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()
    
    def _extract_position(self, state_dict):
        """Extract pipette position from state dictionary."""
        robotId = list(sorted(state_dict.keys()))[0]
        robot_state = state_dict.get(robotId, {})
        position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
        return position

    def _normalize_position(self, position):
        """Normalize position from workspace bounds to [-1, 1]."""
        normalized = 2.0 * (position - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        return normalized.astype(np.float32)