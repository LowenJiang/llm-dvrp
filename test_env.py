#!/usr/bin/env python3
"""
Test script for the DVRP Environment (4D tensor version)

This script demonstrates the environment's state, action, and transition dynamics.
"""

import sys
sys.path.append('src')

import numpy as np
from env import DVRPEnvironment

def test_environment():
    """Test the DVRP environment functionality"""
    
    print("=" * 60)
    print("DVRP Environment Test (4D Tensor)")
    print("=" * 60)
    
    # Create environment with smaller parameters for testing
    env = DVRPEnvironment(
        N=5,  # 5 requests per episode
        num_regions=10,  # 10 spatial regions
        num_time_intervals=12,  # 12 time intervals (6 hours)
        acceptance_rate=0.7  # 70% acceptance rate
    )
    
    print(f"Environment created with:")
    print(f"  - {env.num_regions} spatial regions")
    print(f"  - {env.num_time_intervals} time intervals")
    print(f"  - {env.acceptance_rate} user acceptance rate")
    print(f"  - 4D aggregate tensor: (ℓ, ℓ, h, h)")
    print()
    
    # Reset environment
    state, request = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial request: {request}")
    print()
    
    # Run episode
    total_reward = 0
    for step in range(env.N):
        print(f"--- Step {step + 1} ---")
        
        # Get valid actions
        valid_actions = env.get_valid_actions()
        print(f"Valid actions: {len(valid_actions)}")
        
        # Choose a random valid action
        action = valid_actions[np.random.randint(len(valid_actions))]
        print(f"Chosen action (δ^o, δ^d): {action}")
        
        # Execute action
        next_state, next_request, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"User response (β): {info['user_response']}")
        print(f"Next request: {next_request}")
        print(f"Non-zero entries in state: {np.count_nonzero(state)}")
        
        if done:
            print("Episode finished!")
            break
        print()
    
    # Solve route optimization
    print("=" * 40)
    print("Route Optimization")
    print("=" * 40)
    
    solution = env.solve_route()
    if solution:
        print(f"Route solution:")
        for key, value in solution.items():
            print(f"  {key}: {value}")
    else:
        print("No requests to solve")
    
    print(f"\nTotal reward: {total_reward}")
    print(f"Total requests processed: {len(env.episode_requests)}")
    
    # Show some statistics
    print("\n" + "=" * 40)
    print("Episode Statistics")
    print("=" * 40)
    
    accepted_requests = sum(1 for req in env.episode_requests if req['accepted'])
    print(f"Accepted requests: {accepted_requests}/{len(env.episode_requests)}")
    print(f"Acceptance rate: {accepted_requests/len(env.episode_requests):.2%}")
    
    # Show aggregate tensor statistics
    print(f"State tensor shape: {env.aggregate_tensor.shape}")
    print(f"Non-zero entries: {np.count_nonzero(env.aggregate_tensor)}")
    print(f"Max count in any bin: {np.max(env.aggregate_tensor)}")

if __name__ == "__main__":
    test_environment()
