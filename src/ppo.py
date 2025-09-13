# src/train_sb3_ppo.py
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import os
import sys
from typing import Dict, Any, Tuple, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import RouteEnv
from utils import user


class ActionMaskedRouteEnv(RouteEnv):
    """
    RouteEnv with action masking for Stable Baselines 3 PPO.
    
    This wrapper provides:
    1. Discrete action space for PPO
    2. Action masking based on constraints:
       - Time windows must be within 1-48
       - Origin time must precede destination time
    3. Observation space compatible with SB3
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Convert to discrete action space for PPO
        # Action space: (delta_o, delta_d) where each can be -max_delta to +max_delta
        # Total actions = (2*max_delta + 1)^2
        self.action_dim = (2 * self.max_delta + 1) ** 2
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Keep original observation space but add action mask
        self.observation_space = spaces.Dict({
            "observation": self.observation_space,
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        })
        
        # Action mapping: action_idx -> (delta_o, delta_d)
        self._action_map = {}
        idx = 0
        for delta_o in range(-self.max_delta, self.max_delta + 1):
            for delta_d in range(-self.max_delta, self.max_delta + 1):
                self._action_map[idx] = (delta_o, delta_d)
                idx += 1
    
    def _get_action_mask(self) -> np.ndarray:
        """Get action mask based on current state constraints."""
        mask = np.zeros(self.action_dim, dtype=np.float32)
        
        if self.ptr >= self.N:
            # No more requests, mask all actions
            return mask
        
        # Get current request
        current_req = self.original_reqs[self.ptr]
        o_t_orig = current_req['o_t_index']
        d_t_orig = current_req['d_t_index']
        
        # Check each possible action
        for action_idx, (delta_o, delta_d) in self._action_map.items():
            # Calculate new time windows
            new_o_t = o_t_orig + delta_o
            new_d_t = d_t_orig + delta_d
            
            # Check constraints
            valid = True
            
            # Constraint 1: Time windows must be within 1-48
            if not (1 <= new_o_t <= self.H) or not (1 <= new_d_t <= self.H):
                valid = False
            
            # Constraint 2: Origin time must precede destination time
            if valid and new_o_t >= new_d_t:
                valid = False
            
            # Set mask
            mask[action_idx] = 1.0 if valid else 0.0
        
        return mask
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset environment and return masked observation."""
        obs, info = super().reset(seed=seed, options=options)
        action_mask = self._get_action_mask()
        
        return {
            "observation": obs,
            "action_mask": action_mask
        }, info
    
    def step(self, action):
        """Step environment with action masking."""
        # Convert discrete action to delta tuple
        delta_o, delta_d = self._action_map[action]
        
        # Call parent step with delta tuple
        obs, reward, terminated, truncated, info = super().step([delta_o, delta_d])
        
        # Get action mask for next state
        action_mask = self._get_action_mask()
        
        return {
            "observation": obs,
            "action_mask": action_mask
        }, reward, terminated, truncated, info


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the 4D history aggregate tensor.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract the history aggregate shape
        history_shape = observation_space['observation']['history_aggregate'].shape
        next_req_shape = observation_space['observation']['next_request'].shape
        
        # CNN for 4D history aggregate (L, L, H, H)
        self.history_cnn = nn.Sequential(
            nn.Conv3d(history_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        cnn_output_size = 64 * 4 * 4 * 4  # 64 channels * 4*4*4 spatial
        
        # MLP for next request
        self.next_req_mlp = nn.Sequential(
            nn.Linear(next_req_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Final feature combination
        self.final_mlp = nn.Sequential(
            nn.Linear(cnn_output_size + 32, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract components
        history = observations['observation']['history_aggregate']
        next_req = observations['observation']['next_request']
        
        # Process history with CNN
        history_features = self.history_cnn(history)
        
        # Process next request
        next_req_features = self.next_req_mlp(next_req)
        
        # Combine features
        combined = torch.cat([history_features, next_req_features], dim=-1)
        return self.final_mlp(combined)


class ActionMaskedPPO(PPO):
    """
    PPO with action masking support.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Predict action with action masking.
        """
        if isinstance(observation, dict) and 'action_mask' in observation:
            # Extract observation and action mask
            obs = observation['observation']
            action_mask = observation['action_mask']
            
            # Get action probabilities from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_probs = self.policy.get_distribution(obs_tensor).distribution.probs
                action_probs = action_probs.squeeze(0).cpu().numpy()
            
            # Apply action mask
            masked_probs = action_probs * action_mask
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # If no valid actions, use uniform distribution over valid actions
                masked_probs = action_mask / action_mask.sum()
            
            # Sample action
            if deterministic:
                action = np.argmax(masked_probs)
            else:
                action = np.random.choice(len(masked_probs), p=masked_probs)
            
            return action, state
        else:
            # Fallback to standard prediction
            return super().predict(observation, state, episode_start, deterministic)


def create_env(seed: int = 0) -> ActionMaskedRouteEnv:
    """Create a single environment instance."""
    def _init():
        env = ActionMaskedRouteEnv(
            N=20,
            j=5,
            lambda_penalty=0.1,
            max_delta=6,
            user_function=user.dummy_user,
            seed=seed
        )
        return Monitor(env)
    
    return _init


def train_ppo_sb3(
    total_timesteps: int = 100000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    save_path: str = "models/sb3_ppo_dvrp"
):
    """
    Train PPO agent using Stable Baselines 3.
    """
    
    # Set random seed
    set_random_seed(42)
    
    # Create vectorized environment
    env = make_vec_env(
        create_env,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
        env_kwargs={'seed': 42}
    )
    
    # Create evaluation environment
    eval_env = make_vec_env(
        create_env,
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        env_kwargs={'seed': 123}
    )
    
    # Policy configuration
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
    
    # Create PPO agent
    model = ActionMaskedPPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=batch_size,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print("Starting PPO training with Stable Baselines 3...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of environments: {n_envs}")
    print(f"Action space size: {env.action_space.n}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(save_path + "_final")
    print(f"Training completed! Model saved to {save_path}_final")
    
    return model


def evaluate_model(model_path: str, n_episodes: int = 10):
    """
    Evaluate a trained model.
    """
    # Load model
    model = ActionMaskedPPO.load(model_path)
    
    # Create evaluation environment
    env = create_env(seed=456)()
    
    episode_rewards = []
    episode_costs = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        if 'cost_prev' in info:
            episode_costs.append(info['cost_prev'])
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Cost: {np.mean(episode_costs):.2f} ± {np.std(episode_costs):.2f}")
    
    return episode_rewards, episode_costs


if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Train the model
    model = train_ppo_sb3(
        total_timesteps=100000,
        n_envs=4,
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=10
    )
    
    # Evaluate the model
    print("\nEvaluating trained model...")
    evaluate_model("models/sb3_ppo_dvrp_final", n_episodes=10)
