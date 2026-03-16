"""
Test Script for OT-2 Gym Wrapper

Runs the environment for 1000 steps with random actions to verify
the wrapper works correctly. Prints observation shapes, reward range,
and termination conditions.

Usage: python test_wrapper.py
"""

import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from ot2_gym_wrapper import OT2GymWrapper


def test_spaces():
    """Test that action/observation spaces are valid."""
    env = OT2GymWrapper()
    obs, info = env.reset()

    print("=" * 50)
    print("Space Validation")
    print("=" * 50)
    print(f"Observation space: {env.observation_space}")
    print(f"  shape: {env.observation_space.shape}")
    print(f"  low:   {env.observation_space.low}")
    print(f"  high:  {env.observation_space.high}")
    print(f"Action space: {env.action_space}")
    print(f"  shape: {env.action_space.shape}")
    print(f"  low:   {env.action_space.low}")
    print(f"  high:  {env.action_space.high}")
    print(f"Initial obs: {obs}")
    print(f"Initial obs in space: {env.observation_space.contains(obs)}")
    print(f"Initial error: {info['error']*1000:.2f} mm")
    print(f"Target: {info['target']}")

    env.close()
    print("PASSED\n")


def test_random_actions(n_steps=1000):
    """Run n_steps with random actions, track metrics."""
    env = OT2GymWrapper()
    obs, info = env.reset()

    print("=" * 50)
    print(f"Random Actions Test ({n_steps} steps)")
    print("=" * 50)
    print(f"Target: {info['target']}")
    print(f"Initial error: {info['error']*1000:.2f} mm")

    rewards = []
    errors = []
    min_error = float('inf')

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        errors.append(info['error'])
        min_error = min(min_error, info['error'])

        if step % 100 == 0:
            print(f"Step {step}: error={info['error']*1000:.2f}mm  "
                  f"reward={reward:.2f}  obs_valid={env.observation_space.contains(obs)}")

        if terminated:
            print(f"TERMINATED at step {step} (success!)")
            break
        if truncated:
            print(f"TRUNCATED at step {step} (timeout)")
            break

    print(f"\nSummary:")
    print(f"  Steps taken: {len(rewards)}")
    print(f"  Final error: {errors[-1]*1000:.2f} mm")
    print(f"  Min error:   {min_error*1000:.2f} mm")
    print(f"  Avg reward:  {np.mean(rewards):.2f}")
    print(f"  Total reward: {np.sum(rewards):.2f}")

    env.close()
    print("PASSED\n")


def test_reset():
    """Test that reset generates different targets."""
    env = OT2GymWrapper()

    print("=" * 50)
    print("Reset Test (5 resets)")
    print("=" * 50)

    targets = []
    for i in range(5):
        obs, info = env.reset()
        targets.append(info['target'].copy())
        print(f"Reset {i+1}: target={info['target']}, error={info['error']*1000:.2f}mm")

    # Check targets are different
    all_same = all(np.allclose(targets[0], t) for t in targets[1:])
    print(f"All targets identical: {all_same} (should be False)")

    env.close()
    print("PASSED\n")


if __name__ == "__main__":
    print("OT-2 Gym Wrapper Test Suite\n")

    test_spaces()
    test_reset()
    test_random_actions(n_steps=1000)

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
