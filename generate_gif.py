"""
Generate GIF of RL Agent Reaching Targets
==========================================
Creates a GIF showing the trained RL agent moving to target positions.

Requirements:
    pip install imageio pybullet stable-baselines3

Usage:
    python generate_gif.py --model_path model/best_model --n_episodes 3
"""

import numpy as np
import os
import argparse

try:
    import imageio
except ImportError:
    print("imageio not installed. Run: pip install imageio")
    exit(1)

try:
    import pybullet as p
except ImportError:
    print("PyBullet not installed. Run: pip install pybullet")
    exit(1)

from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env


def generate_gif(model_path, output_path, n_episodes=3, max_steps=100, camera_preset='front', cam_eye=None, cam_target=None):
    """
    Generate GIF of RL agent reaching targets.
    
    Camera presets:
      - front:     View from front
      - side:      View from side  
      - top:       Top-down view
      - isometric: 3D isometric view
      - wide:      Wide angle view
    
    Or use custom coordinates:
      --cam_eye 0.5 -0.5 0.5 --cam_target 0.1 0.1 0.2
    """
    
    # Camera presets - CLEAR FULL ROBOT VIEW
    camera_positions = {
        'front':     {'eye': [0.0, -1.0, 0.6], 'target': [0.0, 0.0, 0.2]},      # Front view, far back
        'side':      {'eye': [1.0, 0.0, 0.6],  'target': [0.0, 0.0, 0.2]},      # Side view
        'top':       {'eye': [0.0, 0.0, 1.5],  'target': [0.0, 0.0, 0.2]},      # Top down
        'isometric': {'eye': [0.8, -0.8, 0.8], 'target': [0.0, 0.0, 0.2]},      # 3D diagonal
        'wide':      {'eye': [1.0, -1.0, 1.0], 'target': [0.0, 0.0, 0.2]},      # Very far back
        'close':     {'eye': [0.3, -0.5, 0.4], 'target': [0.07, 0.05, 0.2]}     # Closer to pipette
    }
    
    # Use custom or preset camera
    if cam_eye and cam_target:
        cam = {'eye': cam_eye, 'target': cam_target}
        print(f"Using custom camera: eye={cam_eye}, target={cam_target}")
    else:
        cam = camera_positions.get(camera_preset, camera_positions['front'])
        print(f"Using camera preset: {camera_preset}")
    print("="*60)
    print("Generating RL Agent GIF")
    print("="*60)
    
    # Create environment with rendering
    print("\nInitializing environment with rendering...")
    env = OT2Env(render=True, max_steps=max_steps)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create frames directory
    frames_dir = "gif_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    
    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1}/{n_episodes} ---")
        obs, _ = env.reset()
        
        # Get goal position for display
        if hasattr(env, 'goal_position'):
            goal = env.goal_position
            print(f"  Target: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")
        
        for step in range(max_steps):
            # Capture frame using PyBullet - HIGH QUALITY
            width, height = 1024, 768  # Higher resolution
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam['eye'],
                cameraTargetPosition=cam['target'],
                cameraUpVector=[0, 0, 1]
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=45,  # Narrower FOV = less distortion, clearer view
                aspect=width / height,
                nearVal=0.01,
                farVal=100.0
            )
            
            _, _, rgb_img, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )
            
            # Save frame
            rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
            imageio.imwrite(frame_path, rgb_array)
            frame_paths.append(frame_path)
            frame_count += 1
            
            # Get action and step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0:
                dist = info.get('distance', 0) * 1000
                print(f"  Step {step}: Distance = {dist:.2f} mm")
            
            if terminated:
                print(f"  ✓ Goal reached at step {step + 1}! Distance: {info['distance']*1000:.3f} mm")
                # Hold on final frame
                for _ in range(15):
                    frame_paths.append(frame_path)
                break
        
        # Add pause between episodes
        if ep < n_episodes - 1:
            for _ in range(10):
                frame_paths.append(frame_path)
    
    env.close()
    
    # Create GIF
    print(f"\nCreating GIF from {len(frame_paths)} frames...")
    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(output_path, images, fps=20, loop=0)
    
    # Cleanup frames
    print("Cleaning up temporary frames...")
    unique_paths = list(set(frame_paths))
    for fp in unique_paths:
        if os.path.exists(fp):
            os.remove(fp)
    if os.path.exists(frames_dir):
        os.rmdir(frames_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ GIF saved to: {output_path}")
    print(f"{'='*60}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate GIF of RL agent')
    parser.add_argument('--model_path', type=str, default='model/best_model',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='rl_agent_demo.gif',
                        help='Output GIF filename')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to record')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Max steps per episode')
    parser.add_argument('--camera', type=str, default='wide',
                        choices=['front', 'side', 'top', 'isometric', 'wide', 'close'],
                        help='Camera angle preset (default: wide for full robot view)')
    parser.add_argument('--cam_eye', type=float, nargs=3, default=None,
                        help='Custom camera eye position [x, y, z]')
    parser.add_argument('--cam_target', type=float, nargs=3, default=None,
                        help='Custom camera target position [x, y, z]')
    args = parser.parse_args()
    
    generate_gif(
        model_path=args.model_path,
        output_path=args.output,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        camera_preset=args.camera,
        cam_eye=args.cam_eye,
        cam_target=args.cam_target
    )


if __name__ == "__main__":
    main()