import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

from env import RouteEnv
from utils import user

def train_ppo():
    """
    Train a PPO agent on the RouteEnv environment.
    """
    # Create the environment
    def make_env():
        return RouteEnv(
            N=10,
            j=5,
            lambda_penalty=0.5,
            max_delta=8,
            user_function=user.dummy_user,
            seed=42,
            max_solve_time=0.1
        )
    
    # Create vectorized environment for training
    env = make_vec_env(make_env, n_envs=2, seed=42)
    
    # Create evaluation environment
    eval_env = Monitor(make_env())
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=2048,     # Reduced from 10000 for more frequent evaluation
        deterministic=True,
        render=False
    )
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",  # For Dict observation space
        env,
        verbose=1,
        tensorboard_log="./ppo_route_tensorboard/",
        learning_rate=3e-4,
        n_steps=256,        # Reduced from 2048 for more frequent updates
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
    total_timesteps = 100000  # Increased to see more progress
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_RouteEnv",
        progress_bar=True    # Show progress bar in terminal
    )
    
    # Save the final model
    model.save("ppo_route_final")
    
    return model

def test_trained_model(model_path="ppo_route_final"):
    """
    Test a trained PPO model on the RouteEnv environment.
    """
    # Create test environment
    env = RouteEnv(
        N=10,
        j=5,
        lambda_penalty=0.1,
        max_delta=8,
        user_function=user.dummy_user,
        seed=42,
        max_solve_time=0.1
    )
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Test the model
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    print("Testing trained model...")
    while steps < 100:  # Test for 100 steps or until done
        action, _states = model.predict(obs, deterministic=True)
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
    
    # Train the model
    print("Starting PPO training...")
    trained_model = train_ppo()
    
    # Test the trained model
    #print("\nTesting the trained model...")
    #final_reward = test_trained_model()
    #print(f"Final test reward: {final_reward:.4f}")
