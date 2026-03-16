"""
PID vs RL Controller Comparison
================================

Runs both controllers on the same 4 targets and produces comparison plots.

Usage: python compare_controllers.py --rl_model models/best_model
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add task10 to path for PID imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task10_pid_controller'))

from stable_baselines3 import PPO, SAC, TD3
from ot2_gym_wrapper import OT2GymWrapper


TARGETS = [
    [0.05, 0.05, 0.18],
    [0.08, 0.03, 0.20],
    [0.03, 0.07, 0.19],
    [0.07, 0.00, 0.17],
]


def load_model(path):
    for cls in [PPO, SAC, TD3]:
        try:
            return cls.load(path)
        except:
            continue
    raise ValueError(f"Cannot load {path}")


def run_rl(model, target, max_steps=500):
    """Run RL controller on a target."""
    env = OT2GymWrapper(max_steps=max_steps, num_substeps=10)
    obs, info = env.reset()
    env.target = np.array(target, dtype=np.float32)
    obs = env._get_obs()

    errors = []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        errors.append(info['error'])
        if terminated or truncated:
            break

    env.close()
    return np.array(errors)


def run_pid(target, max_steps=500):
    """Run PID controller on a target (imports from task10)."""
    try:
        from pid_controller import (PIDController, TUNED_GAINS, OUTPUT_LIMITS,
                                     SUBSTEPS, fix_baseplane, teleport_pipette,
                                     FilteredPosition)
        from sim_class import Simulation
    except ImportError:
        print("  [warn] Could not import PID controller from task10. Skipping PID.")
        return None

    sim = Simulation(num_agents=1, render=False)
    fix_baseplane(sim)
    dt = 0.01 * SUBSTEPS

    pid_x = PIDController(TUNED_GAINS['x'][0], TUNED_GAINS['x'][1], TUNED_GAINS['x'][2], dt, OUTPUT_LIMITS['x'])
    pid_y = PIDController(TUNED_GAINS['y'][0], TUNED_GAINS['y'][1], TUNED_GAINS['y'][2], dt, OUTPUT_LIMITS['y'])
    pid_z = PIDController(TUNED_GAINS['z'][0], TUNED_GAINS['z'][1], TUNED_GAINS['z'][2], dt, OUTPUT_LIMITS['z'])
    pos_filter = FilteredPosition(alpha=0.3)
    target_np = np.array(target)

    teleport_pipette(sim, target[0], target[1], target[2])
    for _ in range(30):
        sim.run([[0, 0, 0, 0]])

    state = sim.run([[0, 0, 0, 0]])
    robot_key = list(state.keys())[0]
    pos = pos_filter.update(state[robot_key]["pipette_position"])

    errors = []
    for _ in range(max_steps):
        vx = pid_x.compute(target_np[0], pos[0])
        vy = pid_y.compute(target_np[1], pos[1])
        vz = pid_z.compute(target_np[2], pos[2])
        state = sim.run([[vx, vy, vz, 0]], num_steps=SUBSTEPS)
        pos = pos_filter.update(state[robot_key]["pipette_position"])
        errors.append(np.linalg.norm(target_np - pos))

    sim.close()
    return np.array(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_model", type=str, default="models/best_model")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("  PID vs RL Controller Comparison")
    print("=" * 60)

    model = load_model(args.rl_model)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    dt = 0.01 * 10
    summary = []

    for i, target in enumerate(TARGETS):
        print(f"\nTarget {i+1}: {target}")

        # RL
        rl_errors = run_rl(model, target)
        rl_final = rl_errors[-1]
        print(f"  RL:  final={rl_final*1000:.2f}mm  steps={len(rl_errors)}")

        # PID
        pid_errors = run_pid(target)
        pid_final = pid_errors[-1] if pid_errors is not None else None
        if pid_final is not None:
            print(f"  PID: final={pid_final*1000:.2f}mm  steps={len(pid_errors)}")

        summary.append({
            "target": target,
            "rl_final": rl_final,
            "pid_final": pid_final,
        })

        # Plot
        ax = axes[i]
        rl_times = np.arange(len(rl_errors)) * dt
        ax.plot(rl_times, rl_errors * 1000, 'b-', lw=1.2, label=f'RL ({rl_final*1000:.1f}mm)')
        if pid_errors is not None:
            pid_times = np.arange(len(pid_errors)) * dt
            ax.plot(pid_times, pid_errors * 1000, 'r-', lw=1.2, label=f'PID ({pid_final*1000:.1f}mm)')
        ax.axhline(1, color='g', ls='--', alpha=0.5, label='1mm')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (mm)')
        ax.set_title(f'Target {i+1}: {target}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('PID vs RL Controller Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/pid_vs_rl_comparison.png', dpi=150)
    plt.close()
    print(f"\nComparison plot: results/pid_vs_rl_comparison.png")

    # Bar chart summary
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TARGETS))
    width = 0.35
    rl_vals = [s['rl_final'] * 1000 for s in summary]
    pid_vals = [s['pid_final'] * 1000 if s['pid_final'] else 0 for s in summary]

    ax.bar(x - width/2, pid_vals, width, label='PID', color='coral')
    ax.bar(x + width/2, rl_vals, width, label='RL', color='steelblue')
    ax.axhline(1, color='g', ls='--', label='1mm target')
    ax.set_xlabel('Target')
    ax.set_ylabel('Final Error (mm)')
    ax.set_title('Final Error Comparison: PID vs RL')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i+1}' for i in range(len(TARGETS))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results/pid_vs_rl_bar.png', dpi=150)
    plt.close()
    print(f"Bar chart: results/pid_vs_rl_bar.png")


if __name__ == "__main__":
    main()
