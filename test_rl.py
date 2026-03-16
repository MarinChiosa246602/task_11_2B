"""
RL Testing Script for OT-2
===========================

Loads a trained model and evaluates it on multiple targets.
Produces per-target plots and a summary comparison.

Usage:
    python test_rl.py                          # uses models/best_model
    python test_rl.py --model models/final_model
    python test_rl.py --model models/best_model --n_episodes 10
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, SAC, TD3
from ot2_gym_wrapper import OT2GymWrapper


def load_model(model_path):
    """Try loading with each algo until one works."""
    for algo_cls in [PPO, SAC, TD3]:
        try:
            return algo_cls.load(model_path)
        except:
            continue
    raise ValueError(f"Could not load model from {model_path}")


def evaluate_episode(model, env, target=None, max_steps=1000):
    """Run one episode, optionally with a fixed target. Returns metrics."""
    obs, info = env.reset()

    # Override target if specified
    if target is not None:
        env.target = np.array(target, dtype=np.float32)
        obs = env._get_obs()

    positions = []
    errors = []
    rewards = []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        positions.append(obs[:3].copy())
        errors.append(info['error'])
        rewards.append(reward)

        if terminated or truncated:
            break

    return {
        "positions": np.array(positions),
        "errors": np.array(errors),
        "rewards": np.array(rewards),
        "final_error": errors[-1],
        "min_error": min(errors),
        "steps": len(errors),
        "target": env.target.copy(),
        "terminated": terminated if 'terminated' in dir() else False,
    }


def plot_episode(result, test_id, save_dir="results"):
    """Plot position and error for one episode."""
    positions = result["positions"]
    errors = result["errors"]
    target = result["target"]
    dt = 0.01 * 10  # substeps
    times = np.arange(len(errors)) * dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, positions[:, 0], 'r-', lw=1.2, label='X')
    ax1.plot(times, positions[:, 1], 'g-', lw=1.2, label='Y')
    ax1.plot(times, positions[:, 2], 'b-', lw=1.2, label='Z')
    ax1.axhline(target[0], color='r', ls='--', alpha=0.5)
    ax1.axhline(target[1], color='g', ls='--', alpha=0.5)
    ax1.axhline(target[2], color='b', ls='--', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title(f'RL Test {test_id} — Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, errors * 1000, 'k-', lw=1)
    ax2.axhline(1, color='g', ls='--', label='1mm')
    ax2.axhline(5, color='orange', ls='--', alpha=0.5, label='5mm')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (mm)')
    ax2.set_title(f'RL Test {test_id} — Error (final: {result["final_error"]*1000:.2f}mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{save_dir}/rl_response_target{test_id}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model")
    parser.add_argument("--n_episodes", type=int, default=5, help="Random target episodes")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("  OT-2 RL Controller — Evaluation")
    print("=" * 60)
    print(f"  Model: {args.model}")

    model = load_model(args.model)
    env = OT2GymWrapper(max_steps=1000, num_substeps=10)

    # ── Fixed targets (same as PID test for comparison) ──────────
    fixed_targets = [
        [0.05, 0.05, 0.18],
        [0.08, 0.03, 0.20],
        [0.03, 0.07, 0.19],
        [0.07, 0.00, 0.17],
    ]

    print("\n--- Fixed Target Tests ---")
    fixed_results = []
    for i, target in enumerate(fixed_targets):
        print(f"\nTest {i+1}: target={target}")
        result = evaluate_episode(model, env, target=target)
        fixed_results.append(result)
        print(f"  Final error: {result['final_error']*1000:.2f} mm")
        print(f"  Min error:   {result['min_error']*1000:.2f} mm")
        print(f"  Steps:       {result['steps']}")
        plot_episode(result, i + 1)

    # ── Random target tests ──────────────────────────────────────
    print(f"\n--- Random Target Tests ({args.n_episodes} episodes) ---")
    random_results = []
    for i in range(args.n_episodes):
        result = evaluate_episode(model, env)
        random_results.append(result)
        print(f"  Episode {i+1}: target={result['target']}  "
              f"final_err={result['final_error']*1000:.2f}mm  "
              f"min_err={result['min_error']*1000:.2f}mm")

    # ── Summary ──────────────────────────────────────────────────
    all_final = [r['final_error'] for r in fixed_results + random_results]
    all_min = [r['min_error'] for r in fixed_results + random_results]

    print(f"\n{'='*60}")
    print(f"  SUMMARY ({len(all_final)} episodes)")
    print(f"{'='*60}")
    print(f"  Mean final error: {np.mean(all_final)*1000:.2f} mm")
    print(f"  Mean min error:   {np.mean(all_min)*1000:.2f} mm")
    print(f"  Median final err: {np.median(all_final)*1000:.2f} mm")
    print(f"  < 1mm:  {sum(1 for e in all_final if e < 0.001)}/{len(all_final)}")
    print(f"  < 5mm:  {sum(1 for e in all_final if e < 0.005)}/{len(all_final)}")
    print(f"  < 10mm: {sum(1 for e in all_final if e < 0.01)}/{len(all_final)}")

    # Save summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    errs_mm = [e * 1000 for e in all_final]
    ax.bar(range(len(errs_mm)), errs_mm, color='steelblue')
    ax.axhline(1, color='g', ls='--', label='1mm target')
    ax.axhline(5, color='orange', ls='--', label='5mm')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Final Error (mm)')
    ax.set_title('RL Controller — Final Error per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/rl_summary.png', dpi=150)
    plt.close()
    print(f"  Summary plot: results/rl_summary.png")

    env.close()


if __name__ == "__main__":
    main()
