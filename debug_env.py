"""
Environment Diagnostic Script
==============================
Tests if the environment is working correctly.
"""

import numpy as np
from ot2_gym_wrapper import OT2Env


def test_environment():
    print("="*60)
    print("ENVIRONMENT DIAGNOSTIC")
    print("="*60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = OT2Env(render=False, max_steps=300)
    print("   ✓ Environment created")
    
    # Test reset
    print("\n2. Testing reset...")
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    
    pipette_pos = obs[:3]
    goal_pos = obs[3:]
    
    print(f"\n   Pipette position: {pipette_pos}")
    print(f"   Goal position:    {goal_pos}")
    
    initial_distance = np.linalg.norm(pipette_pos - goal_pos)
    print(f"   Initial distance: {initial_distance*1000:.2f} mm")
    
    # Test step with zero action
    print("\n3. Testing step with ZERO action [0, 0, 0]...")
    obs2, reward, term, trunc, info = env.step(np.array([0.0, 0.0, 0.0]))
    print(f"   Reward: {reward:.4f}")
    print(f"   New pipette pos: {obs2[:3]}")
    print(f"   Position changed: {not np.allclose(obs[:3], obs2[:3])}")
    
    # Test step moving towards goal
    print("\n4. Testing step moving TOWARDS goal...")
    obs, _ = env.reset()
    pipette_pos = obs[:3]
    goal_pos = obs[3:]
    
    # Calculate direction to goal
    direction = goal_pos - pipette_pos
    direction_normalized = direction / (np.linalg.norm(direction) + 1e-8)
    
    print(f"   Direction to goal: {direction_normalized}")
    
    # Take action towards goal
    action = direction_normalized.astype(np.float32)
    obs2, reward, term, trunc, info = env.step(action)
    
    new_pipette_pos = obs2[:3]
    new_distance = np.linalg.norm(new_pipette_pos - goal_pos)
    old_distance = np.linalg.norm(pipette_pos - goal_pos)
    
    print(f"   Old distance: {old_distance*1000:.2f} mm")
    print(f"   New distance: {new_distance*1000:.2f} mm")
    print(f"   Distance reduced: {new_distance < old_distance}")
    print(f"   Reward: {reward:.4f}")
    
    # Test multiple steps towards goal
    print("\n5. Testing 100 steps moving towards goal...")
    obs, _ = env.reset()
    initial_dist = np.linalg.norm(obs[:3] - obs[3:])
    
    for i in range(100):
        pipette = obs[:3]
        goal = obs[3:]
        direction = goal - pipette
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        action = direction_norm.astype(np.float32)
        obs, reward, term, trunc, info = env.step(action)
        
        if term:
            print(f"   ✓ Goal reached at step {i+1}!")
            break
    
    final_dist = np.linalg.norm(obs[:3] - obs[3:])
    print(f"   Initial distance: {initial_dist*1000:.2f} mm")
    print(f"   Final distance:   {final_dist*1000:.2f} mm")
    print(f"   Improvement:      {(initial_dist - final_dist)*1000:.2f} mm")
    
    # Check if goal is reachable
    print("\n6. Checking workspace bounds...")
    print(f"   Goal X range: 0.0 to 0.15")
    print(f"   Goal Y range: -0.05 to 0.15")
    print(f"   Goal Z range: 0.17 to 0.25")
    print(f"   Current goal: {obs[3:]}")
    
    env.close()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    
    if final_dist < 0.01:
        print("\n✓ Environment appears to be working correctly!")
        print("  The simple controller can reach < 10mm error.")
    else:
        print("\n✗ PROBLEM DETECTED!")
        print(f"  Simple controller only achieved {final_dist*1000:.2f}mm error.")
        print("  There may be an issue with the environment or action scaling.")


if __name__ == "__main__":
    test_environment()
