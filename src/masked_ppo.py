import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os

from env import RouteEnv
from utils import user

def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Action mask function for ActionMasker wrapper.
    Returns valid action mask for the current environment state.
    """
    return env.valid_action_mask()

def train_masked_ppo():
    """
    Train a MaskablePPO agent on the RouteEnv environment with action masking.
    """
    # Create the environment with action masking enabled
    def make_env():
        # Create user function with 50% acceptance rate
        def user_func(rng):
            return user.dummy_user(rng, acceptance_rate=0.5)
        
        base_env = RouteEnv(
            N=10,
            j=5,  # Fixed: j must be < N for environment to work
            lambda_penalty=0.5,
            max_delta=8,
            user_function=user_func,
            seed=42,
            max_solve_time=0.1,
            mask_action=True  # Enable action masking
        )
        # Wrap with ActionMasker
        masked_env = ActionMasker(base_env, mask_fn)
        return masked_env
    
    # Create vectorized environment for training
    env = make_vec_env(make_env, n_envs=2, seed=42)
    
    # Create evaluation environment
    eval_env = Monitor(make_env())
    
    # Create evaluation callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=2048,
        deterministic=True,
        render=False
    )
    
    # Create MaskablePPO model
    model = MaskablePPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for Dict observation spaces
        env,
        verbose=1,
        tensorboard_log="./experiments/",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42
    )
    
    # Train the model
    total_timesteps = 100000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="MaskablePPO_RouteEnv",
        progress_bar=True
    )
    
    # Save the final model
    model.save("maskable_ppo_route_final")
    
    return model

def test_masked_model(model_path="maskable_ppo_route_final"):
    """
    Test a trained MaskablePPO model on the RouteEnv environment.
    """
    # Create test environment
    # Create user function for testing
    def test_user_func(rng):
        return user.dummy_user(rng, acceptance_rate=0.5)
    
    base_env = RouteEnv(
        N=10,
        j=3,  # Fixed: j must be < N
        lambda_penalty=0.1,
        max_delta=8,
        user_function=test_user_func,
        seed=123,  # Different seed for testing
        max_solve_time=0.1,
        mask_action=True
    )
    env = ActionMasker(base_env, mask_fn)
    
    # Load the trained model
    model = MaskablePPO.load(model_path)
    
    # Test the model
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    print("Testing trained masked PPO model...")
    while steps < 100:  # Test for 100 steps or until done
        # Get action masks for current state
        action_masks = get_action_masks(env)
        
        # Predict action with mask
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: Action={action}, Reward={reward:.4f}, Total Reward={total_reward:.4f}")
        
        if terminated or truncated:
            print(f"Episode finished after {steps} steps with total reward: {total_reward:.4f}")
            break
    
    return total_reward

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    # Train the model
    print("Starting MaskablePPO training...")
    trained_model = train_masked_ppo()
    
    # Test the trained model
    print("\nTesting the trained masked model...")
    final_reward = test_masked_model()
    print(f"Final test reward: {final_reward:.4f}")

