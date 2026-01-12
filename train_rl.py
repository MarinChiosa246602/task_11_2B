"""
RL Training Script for OT-2 with ClearML
========================================
Train an RL agent to control the OT-2 pipette positioning.

Usage:
    python train_rl.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --n_epochs 10
"""
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall", "-q"])

import argparse
import numpy as np
from clearml import Task

# =============================================================================
# ClearML Setup - MUST BE BEFORE OTHER IMPORTS
# =============================================================================

# Initialize ClearML task
# NB: Replace 'YourName' with your actual name!
task = Task.init(
    project_name='Mentor Group - Myrthe/Group 2',  # <-- CHANGE 'YourName' TO YOUR NAME
    task_name='PPO_Experiment'
)

# Set the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')

# Set the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

# =============================================================================
# Regular Imports (after ClearML init)
# =============================================================================

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from ot2_gym_wrapper import OT2Env


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for OT-2')
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='PPO',
                        choices=['PPO', 'SAC', 'TD3'])
    
    # Training hyperparameters
    parser.add_argument('--total_timesteps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    
    # Environment
    parser.add_argument('--max_steps', type=int, default=1000)
    
    return parser.parse_args()


# =============================================================================
# Training Function
# =============================================================================

def train(args):
    print("=" * 60)
    print("OT-2 RL Training")
    print("=" * 60)
    print(f"Algorithm: {args.algorithm}")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"N Steps: {args.n_steps}")
    print(f"N Epochs: {args.n_epochs}")
    print("-" * 60)
    
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
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("trained_model")
    print("Model saved to: trained_model.zip")
    
    # Upload model to ClearML
    task.upload_artifact("model", artifact_object="trained_model.zip")
    print("Model uploaded to ClearML artifacts")
    
    # Close environments
    env.close()
    eval_env.close()
    
    print("Training complete!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
