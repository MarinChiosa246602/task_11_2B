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
    task_name='OT2_RL_246602_v2',
)
task.set_base_docker('deanis/2023y2b-rl:latest')

# ============================================================================
# Arguments
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=4096)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--total_timesteps", type=int, default=3000000)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--target_threshold", type=float, default=0.001)
args = parser.parse_args()

task.execute_remotely(queue_name="default")

# ============================================================================
# Curriculum Callback — tightens threshold as agent improves
# ============================================================================
class CurriculumCallback(BaseCallback):
    """
    Curriculum learning:
    - Starts with 5mm threshold (easy)
    - When success rate > 80%, tighten to next level
    - Levels: 5mm → 2mm → 1mm → 0.5mm
    
    Also logs detailed metrics for monitoring.
    """
    
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        
        # Curriculum levels (meters)
        self.levels = [0.005, 0.002, 0.001, 0.0005]
        self.current_level = 0
        self.promotion_threshold = 0.80  # 80% success to advance
        
        # Tracking
        self.episode_successes = []
        self.episode_distances = []
        self.window = 100
    
    def _on_step(self):
        dones = self.locals.get('dones', [])
        for i, done in enumerate(dones):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    dist = info.get('distance_to_goal', float('inf'))
                    best = info.get('best_distance', dist)
                    thresh = self.levels[self.current_level]
                    
                    success = float(dist < thresh)
                    self.episode_successes.append(success)
                    self.episode_distances.append(dist)
                    
                    # Log metrics
                    self.logger.record('ot2/final_distance_mm', dist * 1000)
                    self.logger.record('ot2/best_distance_mm', best * 1000)
                    self.logger.record('ot2/success', success)
                    self.logger.record('ot2/threshold_mm', thresh * 1000)
                    self.logger.record('ot2/curriculum_level', self.current_level)
                    
                    if len(self.episode_successes) >= self.window:
                        recent_rate = np.mean(self.episode_successes[-self.window:])
                        recent_dist = np.mean(self.episode_distances[-self.window:])
                        self.logger.record('ot2/success_rate', recent_rate)
                        self.logger.record('ot2/avg_dist_mm', recent_dist * 1000)
                        
                        # Curriculum promotion
                        if (recent_rate >= self.promotion_threshold and 
                                self.current_level < len(self.levels) - 1):
                            self.current_level += 1
                            new_thresh = self.levels[self.current_level]
                            
                            # Update environment threshold
                            inner_env = self.env
                            while hasattr(inner_env, 'env'):
                                inner_env = inner_env.env
                            inner_env.target_threshold = new_thresh
                            
                            print(f"\n{'='*50}")
                            print(f"  CURRICULUM: Level {self.current_level}")
                            print(f"  New threshold: {new_thresh*1000:.1f}mm")
                            print(f"  Success rate was: {recent_rate*100:.1f}%")
                            print(f"{'='*50}\n")
                            
                            # Reset tracking for new level
                            self.episode_successes = []
                            self.episode_distances = []
        return True
    
    def _on_training_end(self):
        if self.episode_distances:
            print(f"\nFinal curriculum level: {self.current_level}")
            print(f"Final threshold: {self.levels[self.current_level]*1000:.1f}mm")
            print(f"Avg final distance: {1000*np.mean(self.episode_distances[-100:]):.2f}mm")


# ============================================================================
# Training
# ============================================================================
lr_str = f"{args.learning_rate:.0e}".replace("+", "").replace("-0", "-")
run_name = f"lr{lr_str}_b{args.batch_size}_s{args.n_steps}"

print("=" * 60)
print(f"  IMPROVED TRAINING CONFIG")
print(f"  LR: {args.learning_rate}  Batch: {args.batch_size}  Steps: {args.n_steps}")
print(f"  Epochs: {args.n_epochs}  Gamma: {args.gamma}")
print(f"  Total: {args.total_timesteps:,}  Max steps/ep: {args.max_steps}")
print(f"  Curriculum: 5mm → 2mm → 1mm → 0.5mm")
print(f"  Single sim step per action (high resolution)")
print("=" * 60)

# Start with easy threshold — curriculum will tighten it
env = Monitor(OT2Env(
    render=False,
    max_steps=args.max_steps,
    target_threshold=0.005  # Start easy
))

model = PPO(
    'MlpPolicy', env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,         # Small entropy for exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256],   # Larger policy network
            vf=[256, 256],   # Larger value network
        )
    ),
    verbose=1,
    tensorboard_log="runs/",
)

callback = CurriculumCallback(env=env)

# Save checkpoints every 200k steps
save_every = 200000
iterations = max(1, args.total_timesteps // save_every)

for i in range(iterations):
    model.learn(
        total_timesteps=save_every,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name=run_name,
    )
    checkpoint_name = f"{run_name}_{save_every*(i+1)}"
    model.save(checkpoint_name)
    print(f"Checkpoint: {checkpoint_name} (level {callback.current_level})")

model.save(run_name)
task.upload_artifact("model", artifact_object=f"{run_name}.zip")
print(f"\nDone. Final model: {run_name}")
print(f"Curriculum reached level: {callback.current_level}")

try:
    env.close()
except:
    pass