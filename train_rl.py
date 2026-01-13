"""
RL Training Script for OT-2 with ClearML
========================================
FIXED VERSION - Trains properly with goal-relative observations
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
    task_name='PPO_Training_Fixed_v2'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# =============================================================================
# Imports after ClearML
# =============================================================================

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ot2_gym_wrapper import OT2Env


# =============================================================================
# ClearML Logging Callback
# =============================================================================

class ClearMLCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_reward = -np.inf
        self.success_count = 0
        self.logger_clearml = None
        
    def _on_training_start(self):
        self.logger_clearml = Logger.current_logger()
        print("\n" + "="*60)
        print("Training started - FIXED VERSION")
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
                        
                        # Track successes (episode ended early = reached goal)
                        if ep_length < 100:
                            self.success_count += 1
                        
                        # Log to ClearML
                        self.logger_clearml.report_scalar("Episode", "Reward", ep_reward, self.episode_count)
                        self.logger_clearml.report_scalar("Episode", "Length", ep_length, self.episode_count)
                        
                        if len(self.episode_rewards) >= 100:
                            avg = np.mean(self.episode_rewards[-100:])
                            self.logger_clearml.report_scalar("Average", "Reward (100 ep)", avg, self.episode_count)
                            
                            # Success rate
                            recent_successes = sum(1 for l in self.episode_lengths[-100:] if l < 100)
                            success_rate = recent_successes / 100 * 100
                            self.logger_clearml.report_scalar("Average", "Success Rate %", success_rate, self.episode_count)
                        
                        if ep_reward > self.best_reward:
                            self.best_reward = ep_reward
                        
                        if self.episode_count % 100 == 0:
                            avg = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                            avg_len = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
                            print(f"  Ep {self.episode_count:5d} | Reward: {ep_reward:7.1f} | Avg: {avg:7.1f} | Avg Len: {avg_len:5.1f} | Best: {self.best_reward:7.1f}")
        return True
    
    def _on_training_end(self):
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        if self.episode_rewards:
            print(f"Total Episodes: {self.episode_count}")
            print(f"Best Reward: {self.best_reward:.2f}")
            print(f"Final Avg (100): {np.mean(self.episode_rewards[-100:]):.2f}")
            print(f"Final Avg Length: {np.mean(self.episode_lengths[-100:]):.1f}")
            
            # Calculate success rate
            recent_successes = sum(1 for l in self.episode_lengths[-100:] if l < 100)
            print(f"Success Rate (last 100): {recent_successes}%")
        print("="*60)


# =============================================================================
# Training Function
# =============================================================================

def train():
    # Hyperparameters
    total_timesteps = 500000    # 500K should be enough with fixed observations
    learning_rate = 0.0003
    batch_size = 128
    n_steps = 2048
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.01
    max_steps = 100             # Short episodes
    
    print("="*60)
    print("OT-2 RL Training - FIXED VERSION")
    print("="*60)
    print(f"Total Timesteps:   {total_timesteps:,}")
    print(f"Learning Rate:     {learning_rate}")
    print(f"Max Steps/Episode: {max_steps}")
    print(f"Observation:       Goal-relative (10 dim)")
    print("-"*60)
    
    # Log hyperparameters
    task.connect({
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_steps': n_steps,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'max_steps': max_steps,
        'observation_type': 'goal_relative_10dim'
    })
    
    # Create environment
    env = OT2Env(render=False, max_steps=max_steps)
    env = Monitor(env)
    
    eval_env = OT2Env(render=False, max_steps=max_steps)
    eval_env = Monitor(eval_env)
    
    # Create PPO model with tuned network
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])  # Separate policy/value networks
    )
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Callbacks
    callbacks = [
        ClearMLCallback(),
        EvalCallback(
            eval_env, 
            best_model_save_path='./models/', 
            eval_freq=10000,
            n_eval_episodes=20,
            verbose=1
        )
    ]
    
    # Train
    print("\nStarting training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    
    # Save final model
    model.save("trained_model")
    task.upload_artifact("model", artifact_object="trained_model.zip")
    print("Final model saved and uploaded")
    
    # Save best model
    import os
    if os.path.exists("./models/best_model.zip"):
        task.upload_artifact("best_model", artifact_object="./models/best_model.zip")
        print("Best model uploaded")
    
    # Cleanup
    try:
        env.close()
        eval_env.close()
    except:
        pass
    
    print("\nTraining complete!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    train()
