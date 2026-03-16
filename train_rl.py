"""
RL Training Script for OT-2 Pipette Positioning
================================================

Trains a PPO agent using Stable Baselines 3 to control the OT-2
pipette. Supports Weights & Biases logging.

Usage:
    # Basic training
    python train_rl.py

    # With WandB logging
    python train_rl.py --wandb

    # Custom hyperparameters
    python train_rl.py --lr 0.0003 --batch_size 64 --n_steps 2048 --total_timesteps 500000

Author : Marin Chiosa
Course : BUas Applied AI & Data Science – Robotics (OT-2 Simulation)
"""

import argparse
import os
import numpy as np
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from ot2_gym_wrapper import OT2GymWrapper


class MetricsCallback(BaseCallback):
    """Logs training metrics, optionally to WandB."""

    def __init__(self, eval_env, eval_freq=5000, use_wandb=False, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb
        self.best_error = float('inf')

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            errors = []
            for _ in range(5):
                obs, info = self.eval_env.reset()
                episode_error = info['error']
                for _ in range(500):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_error = info['error']
                    if terminated or truncated:
                        break
                errors.append(episode_error)

            mean_error = np.mean(errors)
            min_error = np.min(errors)

            if self.verbose:
                print(f"  [eval @ {self.n_calls}] "
                      f"mean_err={mean_error*1000:.2f}mm  "
                      f"min_err={min_error*1000:.2f}mm  "
                      f"best={self.best_error*1000:.2f}mm")

            if mean_error < self.best_error:
                self.best_error = mean_error
                self.model.save("models/best_model")
                if self.verbose:
                    print(f"    New best! Saved to models/best_model")

            if self.use_wandb:
                import wandb
                wandb.log({
                    "eval/mean_error_mm": mean_error * 1000,
                    "eval/min_error_mm": min_error * 1000,
                    "eval/best_error_mm": self.best_error * 1000,
                    "train/timesteps": self.n_calls,
                })

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent for OT-2")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC", "TD3"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO rollout length")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="ot2-rl-controller")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  OT-2 RL Training")
    print("=" * 60)
    print(f"  Algorithm      : {args.algo}")
    print(f"  Learning rate  : {args.lr}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Gamma          : {args.gamma}")
    print(f"  Max steps/ep   : {args.max_steps}")
    print(f"  Substeps       : {args.substeps}")
    print(f"  Eval freq      : {args.eval_freq}")
    print(f"  WandB          : {args.wandb}")
    print(f"  Seed           : {args.seed}")
    print("=" * 60)

    # WandB init
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.algo}_lr{args.lr}_bs{args.batch_size}_{int(time.time())}",
        )

    # Create environments
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_env = Monitor(
        OT2GymWrapper(max_steps=args.max_steps, num_substeps=args.substeps),
        filename="logs/train"
    )
    eval_env = OT2GymWrapper(max_steps=args.max_steps, num_substeps=args.substeps)

    # Create model
    algo_cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[args.algo]

    if args.algo == "PPO":
        model = algo_cls(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            gamma=args.gamma,
            verbose=1,
            seed=args.seed,
            tensorboard_log="logs/tensorboard",
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    else:
        # SAC / TD3
        model = algo_cls(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1,
            seed=args.seed,
            tensorboard_log="logs/tensorboard",
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    # Callbacks
    metrics_cb = MetricsCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        use_wandb=args.wandb,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/",
        name_prefix="rl_model",
    )

    # Train
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    start_time = time.time()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[metrics_cb, checkpoint_cb],
        progress_bar=True,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Save final model
    model.save("models/final_model")
    print(f"Final model saved to models/final_model")
    print(f"Best model saved to models/best_model")

    if args.wandb:
        import wandb
        wandb.finish()

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
