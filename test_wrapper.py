"""
RL Model Accuracy Test
======================
Evaluate the trained RL model's positioning accuracy.
Measures the final distance error across multiple episodes.
"""

import numpy as np
import argparse
from stable_baselines3 import PPO, SAC, TD3
from ot2_gym_wrapper import OT2Env


def evaluate_model(model_path, algorithm='PPO', n_episodes=100, render=False):
    """
    Evaluate the trained model and measure positioning accuracy.
    
    Args:
        model_path: Path to the trained model (without .zip extension)
        algorithm: Algorithm used (PPO, SAC, TD3)
        n_episodes: Number of episodes to evaluate
        render: Whether to render the simulation
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    # Load the model
    print(f"Loading model: {model_path}")
    if algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'SAC':
        model = SAC.load(model_path)
    elif algorithm == 'TD3':
        model = TD3.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create environment
    env = OT2Env(render=render, max_steps=1000)
    
    # Evaluation metrics
    final_distances = []
    episode_lengths = []
    successes = []
    
    print(f"\nEvaluating on {n_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Get final distance
        pipette_pos = obs[:3]
        goal_pos = obs[3:]
        final_distance = np.linalg.norm(pipette_pos - goal_pos)
        
        final_distances.append(final_distance)
        episode_lengths.append(steps)
        successes.append(final_distance < 0.001)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_dist = np.mean(final_distances[-10:])
            print(f"Episode {episode+1:3d}: Avg Distance (last 10): {avg_dist*1000:.3f} mm")
    
    env.close()
    
    # Calculate metrics
    mean_distance = np.mean(final_distances)
    std_distance = np.std(final_distances)
    min_distance = np.min(final_distances)
    max_distance = np.max(final_distances)
    success_rate = np.mean(successes) * 100
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nPositioning Error Statistics:")
    print(f"  Mean Distance:     {mean_distance*1000:.3f} mm ({mean_distance:.6f} m)")
    print(f"  Std Distance:      {std_distance*1000:.3f} mm")
    print(f"  Min Distance:      {min_distance*1000:.3f} mm")
    print(f"  Max Distance:      {max_distance*1000:.3f} mm")
    print(f"\nPerformance Metrics:")
    print(f"  Success Rate (<1mm): {success_rate:.1f}%")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.1f} steps")
    print("=" * 60)
    
    return {
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'success_rate': success_rate,
        'final_distances': final_distances,
        'episode_lengths': episode_lengths
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL model positioning accuracy')
    parser.add_argument('--model_path', type=str, default='model/best_model',
                        help='Path to trained model (without .zip)')
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the simulation')
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model_path,
        algorithm=args.algorithm,
        n_episodes=args.n_episodes,
        render=args.render
    )
    
    return results


if __name__ == "__main__":
    main()