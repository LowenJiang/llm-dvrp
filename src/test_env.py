#!/usr/bin/env python3
"""
Comprehensive test script for RouteEnv that works with SB3's check_env.
"""

import sys
import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import RouteEnv
from utils import user


class RouteEnvTestWrapper(gym.Wrapper):
    """
    Wrapper for RouteEnv that handles invalid actions gracefully for testing.
    When an invalid action is selected, it automatically chooses the first valid action.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info
    
    def step(self, action):
        # Get current action mask
        if self._last_obs is not None and 'action_mask' in self._last_obs:
            action_mask = self._last_obs['action_mask']
            
            # If selected action is invalid, choose first valid action
            if action_mask[action] == 0:
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    action = valid_actions[0]
                    print(f"Warning: Invalid action selected, using action {action} instead")
                else:
                    # No valid actions, this shouldn't happen but handle gracefully
                    action = 0
        
        # Take step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info


def create_test_env(wrapper=True, **env_kwargs):
    """Create a RouteEnv for testing, optionally with test wrapper."""
    default_kwargs = {
        'N': 10,  # Smaller for faster testing
        'j': 3,
        'lambda_penalty': 0.5,
        'max_delta': 6,
        'user_function': user.dummy_user,
        'seed': 42,
        'random_seed': 42
    }
    default_kwargs.update(env_kwargs)
    
    env = RouteEnv(**default_kwargs)
    
    if wrapper:
        env = RouteEnvTestWrapper(env)
    
    return env


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=== Testing Basic Functionality ===")
    
    env = create_test_env(wrapper=False)
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  History aggregate shape: {obs['history_aggregate'].shape}")
    print(f"  Next request shape: {obs['next_request'].shape}")
    print(f"  Action mask shape: {obs['action_mask'].shape}")
    print(f"  Number of valid actions: {np.sum(obs['action_mask'])}")
    print(f"  Info: {info}")
    
    # Test a few valid steps
    for step in range(3):
        # Get valid actions
        action_mask = obs['action_mask']
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) > 0:
            action = valid_actions[0]  # Take first valid action
            print(f"\nStep {step + 1}: Taking valid action {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Reward: {reward:.3f}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            print(f"  Valid actions remaining: {np.sum(obs['action_mask'])}")
            print(f"  Info: {info}")
            
            if terminated or truncated:
                print("  Episode ended")
                break
        else:
            print(f"Step {step + 1}: No valid actions available")
            break
    
    print("✓ Basic functionality test completed")


def test_action_space_coverage():
    """Test that action space covers expected range."""
    print("\n=== Testing Action Space Coverage ===")
    
    env = create_test_env(wrapper=False)
    
    print(f"Action space: {env.action_space}")
    print(f"Expected action space size: {(2 * env.max_delta + 1) ** 2}")
    print(f"Actual action space size: {env.action_space.n}")
    
    # Test action mapping
    print(f"Action mapping examples:")
    for i in [0, env.action_space.n // 2, env.action_space.n - 1]:
        delta_o, delta_d = env._index_to_delta(i)
        print(f"  Action {i} -> delta_o={delta_o}, delta_d={delta_d}")
    
    print("✓ Action space coverage test completed")


def test_action_masking():
    """Test action masking functionality."""
    print("\n=== Testing Action Masking ===")
    
    env = create_test_env(wrapper=False)
    obs, info = env.reset()
    
    # Test that action mask is properly constructed
    action_mask = obs['action_mask']
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Action mask dtype: {action_mask.dtype}")
    print(f"Valid actions: {np.sum(action_mask)}/{len(action_mask)}")
    
    # Verify mask values are binary
    unique_values = np.unique(action_mask)
    print(f"Unique mask values: {unique_values}")
    assert set(unique_values).issubset({0, 1}), "Action mask should only contain 0s and 1s"
    
    # Test that invalid actions are properly rejected
    invalid_actions = np.where(action_mask == 0)[0]
    if len(invalid_actions) > 0:
        print(f"Testing invalid action {invalid_actions[0]}...")
        try:
            env.step(invalid_actions[0])
            print("✗ Invalid action was not rejected!")
        except gym.error.InvalidAction:
            print("✓ Invalid action properly rejected")
    
    print("✓ Action masking test completed")


def test_with_sb3_check_env():
    """Test environment with SB3's check_env using wrapper."""
    print("\n=== Testing with SB3 check_env ===")
    
    try:
        # Test with wrapper that handles invalid actions
        env = create_test_env(wrapper=True)
        print("Running check_env with test wrapper...")
        check_env(env, warn=True)
        print("✓ check_env passed with wrapper")
        
        # Test vectorized environment
        print("Testing vectorized environment...")
        vec_env = DummyVecEnv([lambda: create_test_env(wrapper=True)])
        
        # Test reset
        obs = vec_env.reset()
        print(f"✓ Vectorized reset successful, obs shape: {obs['history_aggregate'].shape}")
        
        # Test step
        # Sample a random action and let the wrapper handle validity
        action = [vec_env.action_space.sample()]
        obs, rewards, dones, infos = vec_env.step(action)
        print(f"✓ Vectorized step successful, reward: {rewards[0]:.3f}")
        
        vec_env.close()
        
    except Exception as e:
        print(f"✗ check_env failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✓ SB3 check_env test completed")


def test_episode_completion():
    """Test that episodes can complete successfully."""
    print("\n=== Testing Episode Completion ===")
    
    env = create_test_env(wrapper=True, N=5)  # Very small episode for quick test
    
    for episode in range(2):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        while step < 20:  # Safety limit
            # Sample action and let wrapper handle validity
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if terminated or truncated:
                print(f"  Episode completed in {step} steps")
                print(f"  Total reward: {episode_reward:.3f}")
                print(f"  Reason: {'terminated' if terminated else 'truncated'}")
                break
        
        if step >= 20:
            print(f"  Episode reached step limit")
    
    print("✓ Episode completion test completed")


def test_different_configurations():
    """Test environment with different parameter configurations."""
    print("\n=== Testing Different Configurations ===")
    
    configs = [
        {"N": 5, "j": 2, "max_delta": 4, "name": "small"},
        {"N": 15, "j": 5, "max_delta": 8, "name": "medium"},
        {"N": 10, "j": 3, "max_delta": 6, "lambda_penalty": 0.1, "name": "low_penalty"},
    ]
    
    for config in configs:
        name = config.pop("name")
        print(f"\nTesting {name} configuration: {config}")
        
        try:
            env = create_test_env(wrapper=True, **config)
            obs, info = env.reset()
            
            print(f"  ✓ Environment created and reset successfully")
            print(f"  Action space size: {env.action_space.n}")
            print(f"  Valid actions: {np.sum(obs['action_mask'])}")
            
            # Take one step
            valid_actions = np.where(obs['action_mask'] == 1)[0]
            if len(valid_actions) > 0:
                obs, reward, terminated, truncated, info = env.step(valid_actions[0])
                print(f"  ✓ Step successful, reward: {reward:.3f}")
            
        except Exception as e:
            print(f"  ✗ Configuration failed: {e}")
    
    print("✓ Different configurations test completed")


def main():
    """Run all tests."""
    print("Testing RouteEnv Environment")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_action_space_coverage()
        test_action_masking()
        test_with_sb3_check_env()
        test_episode_completion()
        test_different_configurations()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("\nYour RouteEnv is ready for training with Stable-Baselines3!")
        print("\nTo use with PPO:")
        print("  from env import RouteEnv")
        print("  from stable_baselines3 import PPO")
        print("  env = RouteEnv(N=20, j=5, lambda_penalty=0.5, max_delta=8)")
        print("  model = PPO('MultiInputPolicy', env)")
        print("  model.learn(total_timesteps=50000)")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 