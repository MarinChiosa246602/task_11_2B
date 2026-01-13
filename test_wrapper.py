"""
RL Model Test Script (test_wrapper.py)
======================================
Test the trained model and measure positioning accuracy.
Works with the fixed 10-dimensional observation space.
"""

import numpy as np
import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env


def evaluate_model(model_path, n_episodes=100, render=False):
    """Evaluate the trained model."""
    
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    env = OT2Env(render=render, max_steps=100)
    
    distances = []
    lengths = []
    successes = []
    
    print(f"\nEvaluating on {n_episodes} episodes...")
    print("-" * 60)
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Get final distance from info
        final_distance = info['distance']
        
        distances.append(final_distance)
        lengths.append(steps)
        successes.append(final_distance < 0.001)
        
        if (ep + 1) % 10 == 0:
            avg_dist = np.mean(distances[-10:])
            avg_len = np.mean(lengths[-10:])
            print(f"Episode {ep+1:3d}: Avg Distance (last 10): {avg_dist*1000:.3f} mm")
    
    env.close()
    
    # Results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nPositioning Error Statistics:")
    print(f"  Mean Distance:     {np.mean(distances)*1000:.3f} mm ({np.mean(distances):.6f} m)")
    print(f"  Std Distance:      {np.std(distances)*1000:.3f} mm")
    print(f"  Min Distance:      {np.min(distances)*1000:.3f} mm")
    print(f"  Max Distance:      {np.max(distances)*1000:.3f} mm")
    print(f"\nPerformance Metrics:")
    print(f"  Success Rate (<1mm): {np.mean(successes)*100:.1f}%")
    print(f"  Mean Episode Length: {np.mean(lengths):.1f} steps")
    print("=" * 60)
    
    return {
        'mean_distance': np.mean(distances),
        'success_rate': np.mean(successes) * 100,
        'mean_length': np.mean(lengths)
    }


def main():
    parser = argparse.ArgumentParser(description='Test trained RL model')
    parser.add_argument('--model_path', type=str, default='model/best_model',
                        help='Path to trained model (without .zip)')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the simulation')
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.n_episodes, args.render)


if __name__ == "__main__":
    main()