import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task
import argparse
from datetime import datetime
import numpy as np

# Import wrapper
from marin_chiosa_ot2_gym_wrapper import OT2Env
class OT2Callback(BaseCallback):
    
    def __init__(self, threshold=0.005, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_final_distances = []
    
    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        
        for i, done in enumerate(dones):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    final_dist = info.get('distance_to_goal', float('inf'))
                    
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']
                        
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        success = float(final_dist < self.threshold)
                        self.episode_successes.append(success)
                        self.episode_final_distances.append(final_dist)
                        
                        self.logger.record('ot2/episode_reward', ep_reward)
                        self.logger.record('ot2/episode_length', ep_length)
                        self.logger.record('ot2/final_distance_mm', final_dist * 1000)
                        self.logger.record('ot2/success', success)
                        
                        if len(self.episode_successes) >= 10:
                            window = min(100, len(self.episode_successes))
                            self.logger.record('ot2/success_rate_100ep', 
                                             np.mean(self.episode_successes[-window:]))
                            self.logger.record('ot2/avg_length_100ep', 
                                             np.mean(self.episode_lengths[-window:]))
                            self.logger.record('ot2/avg_final_dist_mm_100ep', 
                                             np.mean(self.episode_final_distances[-window:]) * 1000)
        return True
    
    def _on_training_end(self) -> None:
        if len(self.episode_successes) > 0:
            print("\n" + "="*60)
            print("TRAINING SUMMARY")
            print("="*60)
            print(f"Total episodes: {len(self.episode_successes)}")
            print(f"Success rate: {100*np.mean(self.episode_successes):.1f}%")
            print(f"Average episode length: {np.mean(self.episode_lengths):.1f} steps")
            print(f"Average final distance: {1000*np.mean(self.episode_final_distances):.3f} mm")
            print("="*60)


# ============================================================================
# ClearML Setup
# ============================================================================
task = Task.init(
    project_name='Mentor Group - Myrthe/Group 2', 
    task_name=f'ot2_marin_v3',
)

task.set_repo(
    repo='https://github.com/MarinChiosa246602/task_11_2B.git',
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.set_packages(['tensorboard', 'clearml'])

# ============================================================================
# Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=2000000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max_steps_truncate", type=int, default=300)
parser.add_argument("--target_threshold", type=float, default=0.001)
args = parser.parse_args()

# Execute remotely
task.execute_remotely(queue_name='default')

# ============================================================================
# Everything below runs on the REMOTE machine
# ============================================================================

# Simple model name
model_name = f"lr3e-4_b{args.batch_size}_s{args.n_steps}"

print("="*60)
print(f"Training Configuration:")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  Batch Size: {args.batch_size}")
print(f"  N Steps: {args.n_steps}")
print(f"  Total Timesteps: {args.total_timesteps:,}")
print(f"  Max Episode Steps: {args.max_steps_truncate}")
print(f"  Target Threshold: {args.target_threshold*1000:.1f}mm")
print(f"  Model Name: {model_name}")
print("="*60)

# ============================================================================
# Environment Setup
# ============================================================================
env = OT2Env(
    render=False, 
    max_steps=args.max_steps_truncate, 
    target_threshold=args.target_threshold
)

# ============================================================================
# Model Setup
# ============================================================================
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=10,
    gamma=args.gamma,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
)

# ============================================================================
# Training
# ============================================================================
ot2_callback = OT2Callback(threshold=args.target_threshold, verbose=1)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=ot2_callback,
    tb_log_name=f"PPO_{model_name}"
)

# ============================================================================
# Save and Upload Model
# ============================================================================
save_name = f"{model_name}.zip"
model.save(save_name)
print(f"\nModel saved: {save_name}")

task.upload_artifact("model", artifact_object=save_name)
print(f"Artifact uploaded: {save_name}")

print("\nTraining complete!")

try:
    env.close()
except:
    pass