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
    """
    
    def __init__(self, render=False, max_steps=300, target_threshold=0.001):
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
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.steps = 0
        self.goal_position = None
        self.initial_distance = None
    
    def _fix_baseplane(self):
        """Fix the duplicate baseplane URDF that interferes with the robot."""
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
        
        self.goal_position = np.random.uniform(
            self.workspace_low, self.workspace_high
        ).astype(np.float32)
        
        state_dict = self.sim.reset(num_agents=1)
        self._fix_baseplane()
        
        current_pos = self._extract_position(state_dict)
        self.initial_distance = float(np.linalg.norm(current_pos - self.goal_position))
        
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        self.steps = 0
        return observation, {}
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        
        max_velocity = 2.0
        velocity = action * max_velocity
        
        full_action = [float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]
        state_dict = self.sim.run([full_action], num_steps=5)
        
        current_pos = self._extract_position(state_dict)
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
        
        reward = self._calculate_reward(distance_to_goal)
        
        terminated = bool(distance_to_goal < self.target_threshold)
        self.steps += 1
        truncated = bool(self.steps >= self.max_steps)
        
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, distance_to_goal):
        time_penalty = -0.1
        distance_penalty = -10.0 * distance_to_goal
        success_bonus = 50.0 if distance_to_goal < self.target_threshold else 0.0
        return float(time_penalty + distance_penalty + success_bonus)
    
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