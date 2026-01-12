"""
RL Training Script for OT-2 with ClearML
========================================
IMPROVED VERSION - Better hyperparameters for faster convergence
"""
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall", "-q"])

import argparse
import numpy as np
from clearml import Task, Logger

# =============================================================================
# ClearML Setup
# =============================================================================

task = Task.init(
    project_name='Mentor Group - Myrthe/Group 2',
    task_name='PPO_Training_Improved'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# =============================================================================
# Imports after ClearML
# =============================================================================

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ot2_gym_wrapper import OT2Env


# =============================================================================
# ClearML Logging Callback
# =============================================================================

class ClearMLLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_reward = -np.inf
        self.logger_clearml = None
        
    def _on_training_start(self):
        self.logger_clearml = Logger.current_logger()
        print("\n" + "="*60)
        print("Training started")
        print("="*60 + "\n")
        
    def _on_step(self):
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done and 'infos' in self.locals:
                    info = self.locals['infos'][idx]
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']
                        
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        self.episode_count += 1
                        
                        # Log to ClearML
                        self.logger_clearml.report_scalar("Episode Metrics", "Reward", ep_reward, self.episode_count)
                        self.logger_clearml.report_scalar("Episode Metrics", "Length", ep_length, self.episode_count)
                        
                        if len(self.episode_rewards) >= 100:
                            avg = np.mean(self.episode_rewards[-100:])
                            self.logger_clearml.report_scalar("Moving Averages", "Reward (100 ep)", avg, self.episode_count)
                        
                        if ep_reward > self.best_reward:
                            self.best_reward = ep_reward
                        
                        if self.episode_count % 100 == 0:
                            avg = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                            print(f"  Episode {self.episode_count:5d} | Reward: {ep_reward:8.2f} | Avg: {avg:8.2f} | Best: {self.best_reward:8.2f}")
        return True
    
    def _on_training_end(self):
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        if self.episode_rewards:
            print(f"Episodes: {self.episode_count}")
            print(f"Best Reward: {self.best_reward:.2f}")
            print(f"Final Avg (100): {np.mean(self.episode_rewards[-100:]):.2f}")


# =============================================================================
# Arguments - IMPROVED HYPERPARAMETERS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=1000000)  # 1M steps
    parser.add_argument('--learning_rate', type=float, default=0.0003)   # Higher LR
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--max_steps', type=int, default=100)  # Shorter episodes!
    return parser.parse_args()


# =============================================================================
# Training
# =============================================================================

def train(args):
    print("="*60)
    print("OT-2 RL Training - IMPROVED")
    print("="*60)
    print(f"Total Timesteps:   {args.total_timesteps:,}")
    print(f"Learning Rate:     {args.learning_rate}")
    print(f"Max Steps/Episode: {args.max_steps}")
    print("-"*60)
    
    # Log to ClearML
    task.connect(vars(args))
    
    # Create environments with SHORTER episodes
    env = OT2Env(render=False, max_steps=args.max_steps)
    env = Monitor(env)
    
    eval_env = OT2Env(render=False, max_steps=args.max_steps)
    eval_env = Monitor(eval_env)
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1
    )
    
    # Callbacks
    callbacks = [
        ClearMLLoggingCallback(),
        EvalCallback(eval_env, best_model_save_path='./models/', eval_freq=10000, verbose=1)
    ]
    
    # Train
    print("\nStarting training...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
    
    # Save
    model.save("trained_model")
    task.upload_artifact("model", artifact_object="trained_model.zip")
    
    import os
    if os.path.exists("./models/best_model.zip"):
        task.upload_artifact("best_model", artifact_object="./models/best_model.zip")
    
    try:
        env.close()
        eval_env.close()
    except:
        pass
    
    print("\nTraining complete!")


if __name__ == "__main__":
    args = parse_args()
    train(args)