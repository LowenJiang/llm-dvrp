#!/usr/bin/env python3
"""
Example usage of the DVRP Environment

This script demonstrates the key concepts and usage patterns.
"""

import sys
sys.path.append('src')

import numpy as np
from env import DVRPEnvironment

def demonstrate_state_action_transition():
    """Demonstrate the state, action, and transition dynamics"""
    
    print("=" * 60)
    print("DVRP Environment: State, Action, and Transition Demo")
    print("=" * 60)
    
    # Create a small environment for demonstration
    env = DVRPEnvironment(
        N=3,                    # 3 requests per episode
        num_regions=5,          # 5 spatial regions
        num_time_intervals=8,   # 8 time intervals
        max_customers=2,        # Max 2 customers per request
        acceptance_rate=0.6     # 60% acceptance rate
    )
    
    print("1. INITIAL STATE")
    print("-" * 30)
    state, request = env.reset()
    print(f"State tensor shape: {state.shape}")
    print(f"Initial request: {request}")
    print(f"Non-zero entries in state: {np.count_nonzero(state)}")
    print()
    
    print("2. ACTION SPACE")
    print("-" * 30)
    valid_actions = env.get_valid_actions()
    print(f"Number of valid actions: {len(valid_actions)}")
    print(f"Sample actions: {valid_actions[:5]}")
    print(f"Action format: (δ^o, δ^d) where δ^o, δ^d ∈ [-8, 8]")
    print()
    
    print("3. TRANSITION DYNAMICS")
    print("-" * 30)
    
    for step in range(env.N):
        print(f"Step {step + 1}:")
        
        # Get current state
        current_state, current_request = env.get_state()
        print(f"  Current request: {current_request}")
        
        # Get valid actions for current request
        valid_actions = env.get_valid_actions()
        print(f"  Valid actions: {len(valid_actions)}")
        
        # Choose an action
        action = valid_actions[np.random.randint(len(valid_actions))]
        print(f"  Chosen action: {action}")
        
        # Execute action
        next_state, next_request, reward, done, info = env.step(action)
        
        print(f"  User response (β): {info['user_response']}")
        print(f"  Next request: {next_request}")
        print(f"  Non-zero entries in state: {np.count_nonzero(next_state)}")
        
        if done:
            print("  Episode finished!")
        print()
    
    print("4. ROUTE OPTIMIZATION")
    print("-" * 30)
    solution = env.solve_route()
    if solution:
        print(f"Route solution found:")
        print(f"  Status: {solution['status']}")
        print(f"  Total cost: {solution['total_cost']}")
        print(f"  Vehicles used: {solution['num_vehicles_used']}")
        print(f"  Routes: {solution['routes']}")
    else:
        print("No requests to solve")

def demonstrate_aggregate_tensor():
    """Demonstrate how the aggregate tensor works"""
    
    print("\n" + "=" * 60)
    print("Aggregate Tensor Demonstration")
    print("=" * 60)
    
    env = DVRPEnvironment(N=2, num_regions=3, num_time_intervals=4, max_customers=2)
    state, request = env.reset()
    
    print(f"Initial aggregate tensor shape: {state.shape}")
    print(f"Initial tensor (all zeros): {np.count_nonzero(state)} non-zero entries")
    
    # Execute a few actions
    for step in range(2):
        valid_actions = env.get_valid_actions()
        action = valid_actions[np.random.randint(len(valid_actions))]
        next_state, next_request, reward, done, info = env.step(action)
        
        print(f"\nAfter step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  User response: {info['user_response']}")
        print(f"  Non-zero entries: {np.count_nonzero(next_state)}")
        
        # Show which bins were updated
        non_zero_indices = np.where(next_state > 0)
        if len(non_zero_indices[0]) > 0:
            print(f"  Updated bins: {list(zip(*non_zero_indices))}")

if __name__ == "__main__":
    demonstrate_state_action_transition()
    demonstrate_aggregate_tensor()
