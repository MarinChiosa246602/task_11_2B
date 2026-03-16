"""
GIF Generator for RL Controller Demo

Loads the trained model and generates a GIF showing the agent
reaching target positions.

Usage: python generate_gif_rl.py --model models/best_model
"""

import argparse
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pybullet as p
from stable_baselines3 import PPO, SAC, TD3
from ot2_gym_wrapper import OT2GymWrapper
import imageio


def load_model(path):
    for cls in [PPO, SAC, TD3]:
        try:
            return cls.load(path)
        except:
            continue
    raise ValueError(f"Cannot load {path}")


def capture_frame(width=800, height=600):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.0, 0.2],
        distance=0.6, yaw=60, pitch=-30, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=width/height, nearVal=0.1, farVal=100)
    _, _, rgb, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    return np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model")
    parser.add_argument("--n_targets", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("Generating RL Demo GIF...")
    model = load_model(args.model)

    # Use human render mode so pybullet GUI is available for frame capture
    env = OT2GymWrapper(render_mode="human", max_steps=args.max_steps, num_substeps=10)

    frames = []

    targets = [
        [0.05, 0.05, 0.18],
        [0.08, 0.03, 0.20],
        [0.03, 0.07, 0.19],
        [0.07, 0.00, 0.17],
    ][:args.n_targets]

    for i, target in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] Target: {target}")

        obs, info = env.reset()
        env.target = np.array(target, dtype=np.float32)
        obs = env._get_obs()

        # Capture initial frames
        for _ in range(10):
            frames.append(capture_frame())

        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 2 == 0:
                frames.append(capture_frame())

            if step % 50 == 0:
                print(f"  Step {step}: error={info['error']*1000:.2f}mm")

            if terminated or truncated:
                break

        print(f"  Final error: {info['error']*1000:.2f}mm")

        # Pause frames at target
        for _ in range(15):
            frames.append(capture_frame())

    env.close()

    gif_path = 'results/rl_demo.gif'
    imageio.mimsave(gif_path, frames, fps=20, loop=0)
    print(f"\nSaved: {gif_path} ({len(frames)} frames, {os.path.getsize(gif_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
