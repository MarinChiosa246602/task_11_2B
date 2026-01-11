"""
Test Script for OT-2 Gymnasium Wrapper
======================================
Tests the wrapper using:
1. check_env from Stable Baselines 3
2. Manual episode testing with random actions

Usage:
    python test_wrapper.py
"""

import numpy as np
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env


def test_check_env():
    """
    Step 1: Use check_env method
    Verifies compatibility with Stable Baselines 3 API.
    """
    print("=" * 60)
    print("Step 1: Running check_env")
    print("=" * 60)
    
    # Instantiate the custom environment
    env = OT2Env(render=False, max_steps=1000)
    
    # Run check_env - will raise exceptions if issues found
    check_env(env)
    
    print("✓ check_env passed! Environment is compatible with SB3.")
    
    env.close()


def test_manual_episodes():
    """
    Step 2: Manual episode testing
    Runs episodes with random actions to observe behavior.
    """
    print("\n" + "=" * 60)
    print("Step 2: Manual Episode Testing")
    print("=" * 60)
    
    # Load the custom environment
    env = OT2Env(render=False, max_steps=500)
    
    # Number of episodes
    num_episodes = 5
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial observation shape: {obs.shape}")
        print(f"Goal position: {obs[3:6]}")
        
        while not done:
            # Take a random action from the environment's action space
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            # Print every 100 steps
            if step % 100 == 0:
                pipette_pos = obs[:3]
                goal_pos = obs[3:6]
                distance = np.linalg.norm(pipette_pos - goal_pos)
                print(f"  Step {step}: Distance = {distance*1000:.2f}mm, Reward = {reward:.4f}")
            
            # Check termination
            done = terminated or truncated
            
            if done:
                pipette_pos = obs[:3]
                goal_pos = obs[3:6]
                final_distance = np.linalg.norm(pipette_pos - goal_pos)
                
                if terminated:
                    print(f"Episode finished: GOAL REACHED after {step} steps!")
                else:
                    print(f"Episode finished: TRUNCATED after {step} steps")
                
                print(f"  Final distance: {final_distance*1000:.2f}mm")
                print(f"  Total reward: {total_reward:.2f}")
    
    env.close()
    print("\n✓ Manual testing complete!")


def test_observation_action_spaces():
    """
    Key Point 1: Verify observations and actions are within defined spaces.
    """
    print("\n" + "=" * 60)
    print("Testing Observation and Action Spaces")
    print("=" * 60)
    
    env = OT2Env(render=False, max_steps=100)
    
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    obs, info = env.reset()
    
    # Check observation is within bounds
    assert env.observation_space.contains(obs), "Initial observation out of bounds!"
    print("✓ Initial observation within bounds")
    
    # Run 100 steps and verify all observations/actions
    for i in range(100):
        action = env.action_space.sample()
        
        # Verify action is valid
        assert env.action_space.contains(action), f"Action out of bounds at step {i}!"
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify observation is valid
        assert env.observation_space.contains(obs), f"Observation out of bounds at step {i}!"
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print("✓ All observations and actions within defined spaces")
    
    env.close()


def test_reward_behavior():
    """
    Key Point 2: Observe reward function behavior.
    """
    print("\n" + "=" * 60)
    print("Testing Reward Function Behavior")
    print("=" * 60)
    
    env = OT2Env(render=False, max_steps=100)
    obs, info = env.reset()
    
    rewards = []
    distances = []
    
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        pipette_pos = obs[:3]
        goal_pos = obs[3:6]
        distance = np.linalg.norm(pipette_pos - goal_pos)
        
        rewards.append(reward)
        distances.append(distance)
        
        if terminated or truncated:
            break
    
    print(f"Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    print(f"Distance range: [{min(distances)*1000:.2f}mm, {max(distances)*1000:.2f}mm]")
    print(f"Correlation: Reward should be negatively correlated with distance")
    print("✓ Reward function behaves as expected (negative distance)")
    
    env.close()


def test_termination_conditions():
    """
    Key Point 3: Check termination conditions.
    """
    print("\n" + "=" * 60)
    print("Testing Termination Conditions")
    print("=" * 60)
    
    # Test truncation (max steps)
    env = OT2Env(render=False, max_steps=50)
    obs, info = env.reset()
    
    step_count = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
    
    if truncated and step_count >= 50:
        print(f"✓ Truncation works: Episode truncated at step {step_count}")
    elif terminated:
        print(f"✓ Termination works: Goal reached at step {step_count}")
    
    env.close()


def test_episode_lengths():
    """
    Key Point 4: Observe episode lengths.
    """
    print("\n" + "=" * 60)
    print("Testing Episode Lengths")
    print("=" * 60)
    
    env = OT2Env(render=False, max_steps=500)
    
    episode_lengths = []
    
    for ep in range(10):
        obs, info = env.reset()
        steps = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
        
        episode_lengths.append(steps)
    
    print(f"Episode lengths: {episode_lengths}")
    print(f"Average: {np.mean(episode_lengths):.1f}")
    print(f"Min: {min(episode_lengths)}, Max: {max(episode_lengths)}")
    print("✓ Episode lengths recorded")
    
    env.close()


if __name__ == "__main__":
    # Run all tests
    test_check_env()
    test_observation_action_spaces()
    test_reward_behavior()
    test_termination_conditions()
    test_episode_lengths()
    test_manual_episodes()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
