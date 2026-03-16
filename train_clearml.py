"""
ClearML Training Script for OT-2 RL Controller
================================================

This script is designed to run on the ClearML GPU server.
It registers a Task, logs hyperparameters and metrics,
and saves the trained model as an artifact.

Setup (run ONCE on your local machine):
    pip install clearml
    clearml-init  # paste the credentials when prompted

Usage (local — enqueues to GPU server):
    python train_clearml.py

The script will:
  1. Create a ClearML Task
  2. Enqueue it to the GPU worker queue
  3. The GPU server picks it up and trains
  4. You monitor at http://31.204.128.128:8080

Author : Marin Chiosa
"""

import os
import numpy as np
import time

# ── ClearML Setup ────────────────────────────────────────────────────────────
from clearml import Task, Logger

# Create task BEFORE any other imports that might be tracked
task = Task.init(
    project_name="OT2-RL-Controller",
    task_name=f"PPO_training_{int(time.time())}",
    task_type=Task.TaskTypes.training,
)

# Connect hyperparameters (editable from ClearML UI)
params = {
    "algo": "PPO",
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "total_timesteps": 500_000,
    "gamma": 0.99,
    "max_steps_per_episode": 1000,
    "num_substeps": 10,
    "net_arch": [256, 256],
    "seed": 42,
    "eval_freq": 10_000,
}
task.connect(params)

# ── If running remotely, execute on the GPU queue ────────────────────────────
# Uncomment the line below to enqueue instead of running locally:
# task.execute_remotely(queue_name="default", exit_process=True)

# ── Imports (after ClearML init so dependencies are tracked) ─────────────────
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from ot2_gym_wrapper import OT2GymWrapper


# ── ClearML Callback ────────────────────────────────────────────────────────
class ClearMLCallback(BaseCallback):
    """Evaluates the agent periodically and logs to ClearML."""

    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_error = float('inf')
        self.logger_clearml = Task.current_task().get_logger()

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            errors = []
            steps_to_converge = []

            for ep in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                ep_error = info['error']
                converged_step = None

                for step in range(params["max_steps_per_episode"]):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_error = info['error']

                    if converged_step is None and ep_error < 0.001:
                        converged_step = step

                    if terminated or truncated:
                        break

                errors.append(ep_error)
                if converged_step is not None:
                    steps_to_converge.append(converged_step)

            mean_error = np.mean(errors)
            min_error = np.min(errors)
            success_rate = sum(1 for e in errors if e < 0.001) / len(errors)
            mean_converge = np.mean(steps_to_converge) if steps_to_converge else -1

            # Log to ClearML
            self.logger_clearml.report_scalar(
                "eval", "mean_error_mm", mean_error * 1000, self.n_calls)
            self.logger_clearml.report_scalar(
                "eval", "min_error_mm", min_error * 1000, self.n_calls)
            self.logger_clearml.report_scalar(
                "eval", "success_rate", success_rate, self.n_calls)
            self.logger_clearml.report_scalar(
                "eval", "best_error_mm", self.best_error * 1000, self.n_calls)

            if mean_converge > 0:
                self.logger_clearml.report_scalar(
                    "eval", "mean_steps_to_1mm", mean_converge, self.n_calls)

            if self.verbose:
                print(f"  [eval @ {self.n_calls}] "
                      f"mean={mean_error*1000:.2f}mm  "
                      f"min={min_error*1000:.2f}mm  "
                      f"success={success_rate:.0%}  "
                      f"best={self.best_error*1000:.2f}mm")

            # Save best model
            if mean_error < self.best_error:
                self.best_error = mean_error
                self.model.save("models/best_model")
                # Upload to ClearML as artifact
                task.upload_artifact("best_model", "models/best_model.zip")
                if self.verbose:
                    print(f"    New best! Saved & uploaded.")

        return True


# ── Main Training ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  OT-2 RL Training (ClearML)")
    print("=" * 60)
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    # Create environments
    train_env = Monitor(
        OT2GymWrapper(
            max_steps=params["max_steps_per_episode"],
            num_substeps=params["num_substeps"],
        )
    )
    eval_env = OT2GymWrapper(
        max_steps=params["max_steps_per_episode"],
        num_substeps=params["num_substeps"],
    )

    # Create model
    algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
    algo_cls = algo_map[params["algo"]]

    model_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        verbose=1,
        seed=params["seed"],
        policy_kwargs=dict(net_arch=params["net_arch"]),
    )

    if params["algo"] == "PPO":
        model_kwargs["n_steps"] = params["n_steps"]

    model = algo_cls(**model_kwargs)

    # Callback
    clearml_cb = ClearMLCallback(
        eval_env=eval_env,
        eval_freq=params["eval_freq"],
    )

    # Train
    print(f"\nTraining for {params['total_timesteps']} timesteps...")
    start = time.time()

    model.learn(
        total_timesteps=params["total_timesteps"],
        callback=[clearml_cb],
        progress_bar=True,
    )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")

    # Save final model
    model.save("models/final_model")
    task.upload_artifact("final_model", "models/final_model.zip")
    print("Final model saved & uploaded.")

    # Log final summary
    logger = task.get_logger()
    logger.report_scalar("summary", "best_error_mm", clearml_cb.best_error * 1000, 0)
    logger.report_scalar("summary", "training_minutes", elapsed / 60, 0)

    train_env.close()
    eval_env.close()

    print(f"\nBest error: {clearml_cb.best_error*1000:.2f} mm")
    print(f"Monitor at: http://31.204.128.128:8080")


if __name__ == "__main__":
    main()
