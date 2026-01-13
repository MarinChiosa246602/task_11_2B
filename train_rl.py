"""
RL Training Script for OT-2 with ClearML
========================================
4D Normalized Direction Observation
Optimal policy: action = observation[:3]
"""
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall", "-q"])

import numpy as np
from clearml import Task, Logger

# =============================================================================
# ClearML Setup
# =============================================================================

task = Task.init(
    project_name='Mentor Group - Myrthe/Group 2',
    task_name='PPO_NormalizedDir_4D'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# =============================================================================
# Imports
# =============================================================================

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ot2_gym_wrapper import OT2Env


# =============================================================================
# Callback
# =============================================================================

class ClearMLCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_reward = -np.inf
        self.logger_clearml = None
        
    def _on_training_start(self):
        self.logger_clearml = Logger.current_logger()
        print("\n" + "="*60)
        print("Training - 4D NORMALIZED DIRECTION")
        print("Optimal policy: action = observation[:3]")
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
                        
                        self.logger_clearml.report_scalar("Episode", "Reward", ep_reward, self.episode_count)
                        self.logger_clearml.report_scalar("Episode", "Length", ep_length, self.episode_count)
                        
                        if len(self.episode_rewards) >= 100:
                            avg = np.mean(self.episode_rewards[-100:])
                            avg_len = np.mean(self.episode_lengths[-100:])
                            self.logger_clearml.report_scalar("Average", "Reward", avg, self.episode_count)
                            self.logger_clearml.report_scalar("Average", "Length", avg_len, self.episode_count)
                            
                            successes = sum(1 for l in self.episode_lengths[-100:] if l < 100)
                            self.logger_clearml.report_scalar("Average", "Success%", successes, self.episode_count)
                        
                        if ep_reward > self.best_reward:
                            self.best_reward = ep_reward
                        
                        if self.episode_count % 100 == 0:
                            avg = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                            avg_len = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
                            print(f"  Ep {self.episode_count:5d} | Rew: {avg:7.1f} | Len: {avg_len:5.1f} | Best: {self.best_reward:7.1f}")
        return True
    
    def _on_training_end(self):
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        if self.episode_rewards:
            print(f"Episodes: {self.episode_count}")
            print(f"Best Reward: {self.best_reward:.1f}")
            avg_len = np.mean(self.episode_lengths[-100:])
            print(f"Final Avg Length: {avg_len:.1f}")
            successes = sum(1 for l in self.episode_lengths[-100:] if l < 100)
            print(f"Final Success Rate: {successes}%")
        print("="*60)


# =============================================================================
# Training
# =============================================================================

def train():
    # Hyperparameters
    total_timesteps = 500000
    learning_rate = 0.0003
    batch_size = 128
    n_steps = 2048
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.01
    max_steps = 100
    
    print("="*60)
    print("OT-2 RL Training")
    print("="*60)
    print(f"Observation: 4D [dir_x, dir_y, dir_z, distance]")
    print(f"Direction is NORMALIZED (unit vector)")
    print(f"Optimal policy: action = obs[:3]")
    print(f"Total Timesteps: {total_timesteps:,}")
    print("-"*60)
    
    task.connect({
        'total_timesteps': total_timesteps,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_steps': n_steps,
        'n_epochs': n_epochs,
        'max_steps': max_steps,
        'observation': '4D_normalized_direction'
    })
    
    env = OT2Env(render=False, max_steps=max_steps)
    env = Monitor(env)
    
    eval_env = OT2Env(render=False, max_steps=max_steps)
    eval_env = Monitor(eval_env)
    
    # Small network - simple task
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
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
    
    callbacks = [
        ClearMLCallback(),
        EvalCallback(eval_env, best_model_save_path='./models/', eval_freq=10000, verbose=1)
    ]
    
    print("\nStarting training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    
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
    
    print("\nDone!")


if __name__ == "__main__":
    train()
