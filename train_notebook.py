"""
OT-2 RL Training — ClearML Jupyter Notebook Version
=====================================================

Run this in the ClearML remote Jupyter session.

Setup steps:
  1. clearml-session --docker deanis/robosuite:py3.8-2 --queue default --packages "pyvirtualdisplay" "pygame" "box2d" "gym" "matplotlib" "stable-baselines3" "gymnasium" "pybullet" "imageio" "opencv-python" "numpy"
  2. Open Jupyter URL in browser
  3. In terminal: apt install xvfb
  4. Upload all project files to the Jupyter environment
  5. Open this notebook and run cell by cell

Copy the cells below into a Jupyter notebook.
"""

# =============================================================
# CELL 1: Setup virtual display (required for headless pybullet)
# =============================================================
"""
from pyvirtualdisplay import Display
import matplotlib.pyplot as plt

display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display as ipy_display
plt.ion()

print("Virtual display started.")
"""

# =============================================================
# CELL 2: Install any missing packages
# =============================================================
"""
import subprocess
subprocess.run(["pip", "install", "stable-baselines3", "gymnasium", "pybullet", "imageio", "clearml"], check=True)
print("Packages installed.")
"""

# =============================================================
# CELL 3: Setup textures and verify sim works
# =============================================================
"""
import os
import shutil
import numpy as np

os.makedirs('textures/_plates', exist_ok=True)

# Copy texture files if they exist
for f in os.listdir('.'):
    if f.endswith('.png') and (f[0].isdigit() or f.startswith('texture')):
        if not os.path.exists(f'textures/{f}'):
            shutil.copy(f, f'textures/{f}')
        if not os.path.exists(f'textures/_plates/{f}'):
            shutil.copy(f, f'textures/_plates/{f}')

if os.path.exists('uvmapped_dish_large_comp.png'):
    if not os.path.exists('textures/texture1.png'):
        shutil.copy('uvmapped_dish_large_comp.png', 'textures/texture1.png')
    if not os.path.exists('textures/_plates/plate1.png'):
        shutil.copy('uvmapped_dish_large_comp.png', 'textures/_plates/plate1.png')

print("Textures:", os.listdir('textures/'))
print("Plates:", os.listdir('textures/_plates/'))

# Test simulation
from sim_class import Simulation
sim = Simulation(num_agents=1, render=False)
state = sim.run([[0, 0, 0, 0]])
robot_key = list(state.keys())[0]
print(f"Sim OK. Pipette pos: {state[robot_key]['pipette_position']}")
sim.close()
print("Simulation test passed!")
"""

# =============================================================
# CELL 4: Test the gym wrapper
# =============================================================
"""
from ot2_gym_wrapper import OT2GymWrapper

env = OT2GymWrapper()
obs, info = env.reset()
print(f"Obs shape: {obs.shape}")
print(f"Obs: {obs}")
print(f"Target: {info['target']}")
print(f"Error: {info['error']*1000:.2f} mm")

# Run 100 random steps
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 20 == 0:
        print(f"Step {i}: error={info['error']*1000:.2f}mm  reward={reward:.2f}")
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Wrapper test passed!")
"""

# =============================================================
# CELL 5: ClearML Task Init + Training
# =============================================================
"""
import time
import numpy as np
from clearml import Task

# Init ClearML task
task = Task.init(
    project_name="OT2-RL-Controller",
    task_name=f"PPO_train_{int(time.time())}",
    task_type=Task.TaskTypes.training,
)

# Hyperparameters (editable from ClearML UI)
params = {
    "algo": "PPO",
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "total_timesteps": 500_000,
    "gamma": 0.99,
    "max_steps": 1000,
    "substeps": 10,
    "net_arch": [256, 256],
    "seed": 42,
    "eval_freq": 10_000,
}
task.connect(params)

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from ot2_gym_wrapper import OT2GymWrapper

os.makedirs("models", exist_ok=True)

# Eval callback
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_error = float('inf')
        self.logger_cl = task.get_logger()

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            errors = []
            for _ in range(5):
                obs, info = self.eval_env.reset()
                for _ in range(500):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = self.eval_env.step(action)
                    if terminated or truncated:
                        break
                errors.append(info['error'])

            mean_err = np.mean(errors)
            self.logger_cl.report_scalar("eval", "mean_error_mm", mean_err * 1000, self.n_calls)
            self.logger_cl.report_scalar("eval", "min_error_mm", np.min(errors) * 1000, self.n_calls)
            self.logger_cl.report_scalar("eval", "best_error_mm", self.best_error * 1000, self.n_calls)

            print(f"  [eval @ {self.n_calls}] mean={mean_err*1000:.2f}mm  best={self.best_error*1000:.2f}mm")

            if mean_err < self.best_error:
                self.best_error = mean_err
                self.model.save("models/best_model")
                task.upload_artifact("best_model", "models/best_model.zip")
                print(f"    New best! Saved.")
        return True

# Create envs
train_env = Monitor(OT2GymWrapper(max_steps=params["max_steps"], num_substeps=params["substeps"]))
eval_env = OT2GymWrapper(max_steps=params["max_steps"], num_substeps=params["substeps"])

# Create model
algo_cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[params["algo"]]
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

eval_cb = EvalCallback(eval_env, eval_freq=params["eval_freq"])

print(f"Training {params['algo']} for {params['total_timesteps']} steps...")
start = time.time()

model.learn(total_timesteps=params["total_timesteps"], callback=[eval_cb])

elapsed = time.time() - start
print(f"Done in {elapsed/60:.1f} min. Best error: {eval_cb.best_error*1000:.2f}mm")

model.save("models/final_model")
task.upload_artifact("final_model", "models/final_model.zip")

train_env.close()
eval_env.close()
print("Training complete!")
"""

# =============================================================
# CELL 6: Evaluate the trained model
# =============================================================
"""
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2GymWrapper
import numpy as np

model = PPO.load("models/best_model")
env = OT2GymWrapper(max_steps=1000, num_substeps=10)

targets = [
    [0.05, 0.05, 0.18],
    [0.08, 0.03, 0.20],
    [0.03, 0.07, 0.19],
    [0.07, 0.00, 0.17],
]

print("Evaluating on fixed targets...")
for i, target in enumerate(targets):
    obs, info = env.reset()
    env.target = np.array(target, dtype=np.float32)
    obs = env._get_obs()
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"  Target {i+1} {target}: error={info['error']*1000:.2f}mm  steps={step+1}")

env.close()
"""

# =============================================================
# CELL 7: Visualize results
# =============================================================
"""
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2GymWrapper
import numpy as np

model = PPO.load("models/best_model")
env = OT2GymWrapper(max_steps=500, num_substeps=10)

target = [0.05, 0.05, 0.18]
obs, info = env.reset()
env.target = np.array(target, dtype=np.float32)
obs = env._get_obs()

positions = []
errors = []

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    positions.append(obs[:3].copy())
    errors.append(info['error'])
    if terminated or truncated:
        break

env.close()

positions = np.array(positions)
dt = 0.01 * 10
times = np.arange(len(errors)) * dt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(times, positions[:, 0], 'r-', label='X')
ax1.plot(times, positions[:, 1], 'g-', label='Y')
ax1.plot(times, positions[:, 2], 'b-', label='Z')
ax1.axhline(target[0], color='r', ls='--', alpha=0.5)
ax1.axhline(target[1], color='g', ls='--', alpha=0.5)
ax1.axhline(target[2], color='b', ls='--', alpha=0.5)
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Position (m)')
ax1.set_title('RL Agent — Position vs Time')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(times, np.array(errors) * 1000, 'k-')
ax2.axhline(1, color='g', ls='--', label='1mm')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (mm)')
ax2.set_title(f'RL Agent — Error (final: {errors[-1]*1000:.2f}mm)')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/rl_response.png', dpi=150)
plt.show()
print("Plot saved!")
"""
