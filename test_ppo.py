#!/usr/bin/env python3
"""
Test script for PPO implementation
"""

import sys
import os
sys.path.append('src')

import numpy as np
import torch
from src.models.ppo import PPOAgent, train_ppo
from src.env import DVRPEnvironment
from src.route_class import Route

def test_ppo_basic():
    """Test basic PPO functionality"""
    print("Testing PPO Basic Functionality...")
    
    # Create environment
    route = Route(N=5, auto_solve=False)
    env = DVRPEnvironment(route, acceptance_rate=0.7, alpha=2.0)
    
    # Get state dimensions
    state, _ = env.reset()
    state_dim = state.shape
    action_dim = 95 * 95  # 95 x 95 action space
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create PPO agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,  # Smaller for testing
        learning_rate=3e-4,
        device=device
    )
    
    # Test single episode
    print("\nTesting single episode...")
    state, request = env.reset()
    episode_reward = 0
    step = 0
    max_steps = 5
    
    while step < max_steps:
        valid_actions = env.get_valid_actions()
        print(f"Step {step}: {len(valid_actions)} valid actions")
        
        if not valid_actions:
            break
        
        # Get action from agent
        action_idx, log_prob, value = agent.get_action(state, valid_actions)
        action = valid_actions[action_idx]
        
        print(f"  Action: {action}, Log prob: {log_prob:.3f}, Value: {value:.3f}")
        
        # Execute action
        next_state, next_request, reward, done, info = env.step(action)
        
        print(f"  Reward: {reward:.3f}, User response: {info['user_response']}")
        
        # Store transition
        agent.store_transition(state, action_idx, reward, next_state, done, log_prob, value)
        
        state = next_state
        episode_reward += reward
        step += 1
        
        if done:
            break
    
    print(f"Episode reward: {episode_reward:.3f}")
    print(f"Final cost: {env.route.cost['total_cost'] if env.route.cost else 'N/A'}")
    
    # Test update
    print("\nTesting policy update...")
    agent.update()
    print("Policy update completed successfully!")
    
    print("\nBasic PPO test completed successfully!")

def test_ppo_training():
    """Test PPO training with a few episodes"""
    print("\nTesting PPO Training...")
    
    # Create environment
    route = Route(N=10, auto_solve=False)
    env = DVRPEnvironment(route, acceptance_rate=0.7, alpha=2.0)
    
    # Get state dimensions
    state, _ = env.reset()
    state_dim = state.shape
    action_dim = 95 * 95
    
    # Create PPO agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=3e-4,
        device=device
    )
    
    # Train for a few episodes
    print("Training for 5 episodes...")
    episode_rewards, episode_costs = train_ppo(
        env=env,
        agent=agent,
        num_episodes=5,
        max_steps_per_episode=10,
        update_frequency=8,
        save_frequency=10
    )
    
    print(f"Training completed!")
    print(f"Episode rewards: {episode_rewards}")
    print(f"Episode costs: {episode_costs}")

if __name__ == "__main__":
    test_ppo_basic()
    test_ppo_training()
