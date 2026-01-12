import numpy as np
from ot2_gym_wrapper import OT2Env


class SimpleController:
    """
    Simple proportional controller that moves towards the goal.
    """
    
    def __init__(self, gain=1.0):
        self.gain = gain
    
    def predict(self, obs, deterministic=True):
        """
        Predict action to move towards goal.
        Compatible with SB3 model interface.
        """
        pipette_pos = obs[:3]
        goal_pos = obs[3:]
        
        # Calculate direction to goal
        direction = goal_pos - pipette_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.001:  # If not at goal
            # Normalize and scale
            action = (direction / distance) * self.gain
            # Clip to action space [-1, 1]
            action = np.clip(action, -1.0, 1.0)
        else:
            action = np.zeros(3)
        
        return action.astype(np.float32), None


def evaluate_simple_controller(n_episodes=100, render=False):
    """
    Evaluate the simple controller.
    """
    print("="*60)
    print("Simple Controller Evaluation")
    print("="*60)
    
    # Create environment and controller
    env = OT2Env(render=render, max_steps=300)
    controller = SimpleController(gain=1.0)
    
    # Metrics
    distances = []
    steps_list = []
    successes = []
    
    print(f"\nEvaluating on {n_episodes} episodes...")
    print("-"*60)
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = controller.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Final distance
        final_distance = np.linalg.norm(obs[:3] - obs[3:])
        distances.append(final_distance)
        steps_list.append(steps)
        successes.append(final_distance < 0.001)
        
        if (ep + 1) % 10 == 0:
            avg_dist = np.mean(distances[-10:])
            print(f"Episode {ep+1:3d}: Avg Distance (last 10): {avg_dist*1000:.3f} mm")
    
    env.close()
    
    # Results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nPositioning Error Statistics:")
    print(f"  Mean Distance:     {np.mean(distances)*1000:.3f} mm ({np.mean(distances):.6f} m)")
    print(f"  Std Distance:      {np.std(distances)*1000:.3f} mm")
    print(f"  Min Distance:      {np.min(distances)*1000:.3f} mm")
    print(f"  Max Distance:      {np.max(distances)*1000:.3f} mm")
    print(f"\nPerformance Metrics:")
    print(f"  Success Rate (<1mm): {np.mean(successes)*100:.1f}%")
    print(f"  Mean Episode Steps:  {np.mean(steps_list):.1f}")
    print("="*60)
    
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'success_rate': np.mean(successes) * 100,
        'distances': distances
    }


if __name__ == "__main__":
    evaluate_simple_controller(n_episodes=100, render=False)
