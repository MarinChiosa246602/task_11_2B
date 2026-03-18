import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import argparse
import os
from clearml import Task

from marin_chiosa_ot2_gym_wrapper import OT2Env

# ============================================================================
# ClearML Setup
# ============================================================================
task = Task.init(
    project_name='Mentor Group - Myrthe/Group 2',
    task_name='OT2_RL_246602',
)

task.set_base_docker('deanis/2023y2b-rl:latest')

# ============================================================================
# Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--total_timesteps", type=int, default=2000000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--target_threshold", type=float, default=0.001)
args = parser.parse_args()

# Send to GPU server
task.execute_remotely(queue_name="default")

# ============================================================================
# Callback
# ============================================================================
class OT2Callback(BaseCallback):
    def __init__(self, threshold=0.001, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.episode_successes = []
        self.episode_final_distances = []

    def _on_step(self):
        dones = self.locals.get('dones', [])
        for i, done in enumerate(dones):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    dist = info.get('distance_to_goal', float('inf'))
                    self.episode_successes.append(float(dist < self.threshold))
                    self.episode_final_distances.append(dist)

                    self.logger.record('ot2/final_distance_mm', dist * 1000)
                    self.logger.record('ot2/success', float(dist < self.threshold))

                    if len(self.episode_successes) >= 10:
                        w = min(100, len(self.episode_successes))
                        self.logger.record('ot2/success_rate', np.mean(self.episode_successes[-w:]))
                        self.logger.record('ot2/avg_dist_mm', np.mean(self.episode_final_distances[-w:]) * 1000)
        return True

    def _on_training_end(self):
        if self.episode_successes:
            print(f"\nSuccess rate: {100*np.mean(self.episode_successes):.1f}%")
            print(f"Avg final distance: {1000*np.mean(self.episode_final_distances):.2f} mm")

# ============================================================================
# Training
# ============================================================================
lr_str = f"{args.learning_rate:.0e}".replace("+", "").replace("-0", "-")
run_name = f"lr{lr_str}_b{args.batch_size}_s{args.n_steps}"

print("="*60)
print(f"  LR: {args.learning_rate}  Batch: {args.batch_size}  Steps: {args.n_steps}")
print(f"  Epochs: {args.n_epochs}  Gamma: {args.gamma}  Total: {args.total_timesteps:,}")
print(f"  Max steps/ep: {args.max_steps}  Threshold: {args.target_threshold*1000:.1f}mm")
print("="*60)

env = Monitor(OT2Env(render=False, max_steps=args.max_steps, target_threshold=args.target_threshold))

model = PPO(
    'MlpPolicy', env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="runs/",
)

callback = OT2Callback(threshold=args.target_threshold)

save_every = 100000
iterations = max(1, args.total_timesteps // save_every)

for i in range(iterations):
    model.learn(
        total_timesteps=save_every,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name=run_name,
    )
    model.save(f"{run_name}_{save_every*(i+1)}")
    print(f"Checkpoint: {run_name}_{save_every*(i+1)}")

model.save(run_name)
task.upload_artifact("model", artifact_object=f"{run_name}.zip")
print(f"Done. Model: {run_name}")

try:
    env.close()
except:
    pass