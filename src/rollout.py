import numpy as np
from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from env import RouteEnv
from utils import user

def mask_fn(env):
    """Action mask function for ActionMasker wrapper."""
    return env.valid_action_mask()

def rollout_best_model():
    """
    Load the best model from logs and print the course of actions.
    """
    # Create user function
    def user_func(rng):
        return user.dummy_user(rng, acceptance_rate=0.5)
    
    # Create base environment with masking enabled
    base_env = RouteEnv(
        N=10,
        j=5,
        lambda_penalty=0.5,
        max_delta=8,
        user_function=user_func,
        seed=42,
        max_solve_time=0.1,
        mask_action=True  # Enable masking for MaskablePPO
    )
    
    # Wrap with ActionMasker
    env = ActionMasker(base_env, mask_fn)
    
    # Load the best model as MaskablePPO
    print("Loading best model from logs/best_model.zip...")
    model = MaskablePPO.load("logs/best_model.zip")
    
    # Run rollout
    print("Starting rollout...")
    obs, info = env.reset()
    total_reward = 0
    step = 0
    
    print(f"Initial state: {base_env.N - base_env.ptr} requests remaining")
    print("=" * 60)
    
    while step < 100:  # Safety limit
        # Get action masks for MaskablePPO
        action_masks = get_action_masks(env)
        
        # Get action from model
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=False)
        
        # Convert action to integer if it's an array
        action_idx = int(action) if hasattr(action, '__iter__') else int(action)
        
        # Convert action to deltas for display
        if action_idx < len(base_env._action_map):
            delta_o, delta_d = base_env._action_map[action_idx]
        else:
            delta_o, delta_d = 0, 0
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Print action details
        print(f"Step {step:2d}: Action={action_idx:3d} -> deltas=({delta_o:+2d},{delta_d:+2d}) | "
              f"Reward={reward:8.2f} | Accepted={str(info['accepted']):5s} | "
              f"Penalty={info['penalty']:5.2f} | Total={total_reward:8.2f}")
        
        if terminated or truncated:
            print("=" * 60)
            print(f"Episode finished after {step} steps")
            print(f"Final total reward: {total_reward:.2f}")
            break
    
    return total_reward

if __name__ == "__main__":
    rollout_best_model()
