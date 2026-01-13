"""
Test Environment - Verify 4D normalized wrapper works
=====================================================
"""

import numpy as np
from ot2_gym_wrapper import OT2Env


def test_environment():
    print("="*60)
    print("TESTING 4D NORMALIZED DIRECTION WRAPPER")
    print("="*60)
    
    env = OT2Env(render=False, max_steps=100)
    
    # Test 1: Check observation
    print("\n1. Observation Check:")
    obs, _ = env.reset()
    print(f"   Shape: {obs.shape} (should be 4)")
    print(f"   obs[:3] (direction): {obs[:3]}")
    print(f"   obs[3] (distance):   {obs[3]:.4f}")
    
    # Check direction is normalized
    direction_norm = np.linalg.norm(obs[:3])
    print(f"   Direction magnitude: {direction_norm:.4f} (should be ~1.0)")
    
    # Test 2: If action = direction, we should get closer
    print("\n2. Test action = observation[:3]:")
    obs, _ = env.reset()
    initial_distance = obs[3] * 0.3  # Unnormalize
    
    for i in range(50):
        action = obs[:3].copy()  # Action = normalized direction
        obs, reward, term, trunc, info = env.step(action)
        
        if i < 5 or term:
            print(f"   Step {i+1}: dist={info['distance']*1000:.2f}mm, reward={reward:.1f}")
        
        if term:
            print(f"   ✓ Goal reached at step {i+1}!")
            break
    
    final_distance = info['distance']
    print(f"\n   Initial: {initial_distance*1000:.2f}mm → Final: {final_distance*1000:.3f}mm")
    
    env.close()
    
    print("\n" + "="*60)
    if final_distance < 0.001:
        print("✓ SUCCESS! Environment works correctly.")
        print("  action = obs[:3] reaches goal < 1mm")
    else:
        print(f"✗ Did not reach goal. Final: {final_distance*1000:.2f}mm")
    print("="*60)


if __name__ == "__main__":
    test_environment()
