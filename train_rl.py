"""
RL Training Script for OT-2 with ClearML
========================================
Train an RL agent to control the OT-2 pipette positioning.
WITH FULL TRAINING PROGRESS LOGGING

Usage:
    python train_rl.py
"""
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall", "-q"])

import argparse
import numpy as np
from clearml import Task, Logger

# =============================================================================
# ClearML Setup - MUST BE BEFORE OTHER IMPORTS
# =============================================================================

# Initialize ClearML task
task = Task.init(
    project_name='Mentor Group - Myrthe/Group 2',
    task_name='PPO_Training_3M_Steps'
)

# Set the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')

# Set the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

# =============================================================================
# Regular Imports (after ClearML init)
# =============================================================================

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ot2_gym_wrapper import OT2Env


# =============================================================================
# Custom Callback for ClearML Logging
# =============================================================================

class ClearMLLoggingCallback(BaseCallback):
    """
    Custom callback that logs training metrics to ClearML.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_reward = -np.inf
        self.logger_clearml = None
        
    def _on_training_start(self) -> None:
        self.logger_clearml = Logger.current_logger()
        print("\n" + "="*60)
        print("Training started - Logging to ClearML")
        print("="*60 + "\n")
        
    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    if 'infos' in self.locals:
                        info = self.locals['infos'][idx]
                        if 'episode' in info:
                            ep_reward = info['episode']['r']
                            ep_length = info['episode']['l']
                            
                            self.episode_rewards.append(ep_reward)
                            self.episode_lengths.append(ep_length)
                            self.episode_count += 1
                            
                            # Log to ClearML
                            self.logger_clearml.report_scalar(
                                title="Episode Metrics",
                                series="Reward",
                                value=ep_reward,
                                iteration=self.episode_count
                            )
                            self.logger_clearml.report_scalar(
                                title="Episode Metrics",
                                series="Length",
                                value=ep_length,
                                iteration=self.episode_count
                            )
                            
                            # Moving averages
                            if len(self.episode_rewards) >= 10:
                                avg_reward_10 = np.mean(self.episode_rewards[-10:])
                                self.logger_clearml.report_scalar(
                                    title="Moving Averages",
                                    series="Reward (10 ep)",
                                    value=avg_reward_10,
                                    iteration=self.episode_count
                                )
                            
                            if len(self.episode_rewards) >= 100:
                                avg_reward_100 = np.mean(self.episode_rewards[-100:])
                                self.logger_clearml.report_scalar(
                                    title="Moving Averages",
                                    series="Reward (100 ep)",
                                    value=avg_reward_100,
                                    iteration=self.episode_count
                                )
                            
                            # Track best reward
                            if ep_reward > self.best_reward:
                                self.best_reward = ep_reward
                                self.logger_clearml.report_scalar(
                                    title="Best Metrics",
                                    series="Best Reward",
                                    value=self.best_reward,
                                    iteration=self.episode_count
                                )
                            
                            # Print progress every 100 episodes
                            if self.episode_count % 100 == 0:
                                avg = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards[-10:])
                                print(f"  Episode {self.episode_count:5d} | "
                                      f"Reward: {ep_reward:8.2f} | "
                                      f"Avg: {avg:8.2f} | "
                                      f"Best: {self.best_reward:8.2f} | "
                                      f"Timesteps: {self.num_timesteps:,}")
        
        return True
    
    def _on_training_end(self) -> None:
        print("\n" + "="*60)
        print("TRAINING COMPLETE - FINAL STATISTICS")
        print("="*60)
        
        if len(self.episode_rewards) > 0:
            print(f"Total Episodes:        {self.episode_count}")
            print(f"Total Timesteps:       {self.num_timesteps:,}")
            print(f"Best Reward:           {self.best_reward:.2f}")
            print(f"Mean Reward:           {np.mean(self.episode_rewards):.2f}")
            
            if len(self.episode_rewards) >= 100:
                print(f"Final Avg (last 100):  {np.mean(self.episode_rewards[-100:]):.2f}")
            
            # Log final summary
            self.logger_clearml.report_single_value("Final/Total Episodes", self.episode_count)
            self.logger_clearml.report_single_value("Final/Best Reward", self.best_reward)
            self.logger_clearml.report_single_value("Final/Mean Reward", np.mean(self.episode_rewards))
        
        print("="*60 + "\n")


# =============================================================================
# Argument Parser with YOUR HYPERPARAMETERS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for OT-2')
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'SAC', 'TD3'])
    
    # YOUR HYPERPARAMETERS
    parser.add_argument('--total_timesteps', type=int, default=3000000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_threshold', type=float, default=0.001)
    parser.add_argument('--max_steps', type=int, default=300)
    
    # Additional PPO parameters
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    
    return parser.parse_args()


# =============================================================================
# Training Function
# =============================================================================

def train(args):
    print("=" * 60)
    print("OT-2 RL Training")
    print("=" * 60)
    print(f"Algorithm:         {args.algorithm}")
    print(f"Total Timesteps:   {args.total_timesteps:,}")
    print(f"Learning Rate:     {args.learning_rate}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"N Steps:           {args.n_steps}")
    print(f"Gamma:             {args.gamma}")
    print(f"Target Threshold:  {args.target_threshold} m ({args.target_threshold*1000} mm)")
    print(f"Max Steps/Episode: {args.max_steps}")
    print("-" * 60)
    
    # Log hyperparameters to ClearML
    task.connect({
        'algorithm': args.algorithm,
        'total_timesteps': args.total_timesteps,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'n_steps': args.n_steps,
        'gamma': args.gamma,
        'target_threshold': args.target_threshold,
        'max_steps': args.max_steps,
        'n_epochs': args.n_epochs,
        'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_range,
        'ent_coef': args.ent_coef
    })
    
    # Create training environment
    env = OT2Env(render=False, max_steps=args.max_steps)
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = OT2Env(render=False, max_steps=args.max_steps)
    eval_env = Monitor(eval_env)
    
    # Create model
    if args.algorithm == 'PPO':
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
    elif args.algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1
        )
    elif args.algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1
        )
    
    # Create callbacks
    clearml_callback = ClearMLLoggingCallback()
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    callbacks = [clearml_callback, eval_callback]
    
    # Train
    print("\nStarting training...")
    print(f"This will take a while (~{args.total_timesteps//100000} minutes estimated)")
    print("Progress will be logged to ClearML SCALARS tab\n")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save("trained_model")
    print("\nModel saved to: trained_model.zip")
    
    # Upload model to ClearML
    task.upload_artifact("model", artifact_object="trained_model.zip")
    print("Model uploaded to ClearML artifacts")
    
    # Upload best model if exists
    import os
    best_model_path = "./models/best_model.zip"
    if os.path.exists(best_model_path):
        task.upload_artifact("best_model", artifact_object=best_model_path)
        print("Best model uploaded to ClearML artifacts")
    
    # Close environments safely
    try:
        env.close()
    except:
        pass
    try:
        eval_env.close()
    except:
        pass
    
    print("\nTraining complete!")
    print("View results in ClearML: SCALARS tab for graphs, ARTIFACTS for models")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)