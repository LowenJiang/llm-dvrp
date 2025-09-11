import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DVRPEnvironment
from route_class import Route


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for DVRP Environment
    
    This implementation includes:
    - Actor-Critic network architecture
    - PPO loss with clipping
    - Experience buffer for training
    - Action space handling for discrete actions
    """
    
    def __init__(self, 
                 state_dim: Tuple[int, ...],
                 action_dim: int,
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Initialize networks
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.memory = PPOMemory()
        
    def get_action(self, state: np.ndarray, valid_actions: List[Tuple[int, int]]) -> Tuple[int, float, float]:
        """
        Get action from current policy
        
        Parameters:
        -----------
        state: np.ndarray
            Current state (4D tensor)
        valid_actions: List[Tuple[int, int]]
            List of valid actions for current state
            
        Returns:
        --------
        action_idx: int
            Index of selected action in valid_actions
        log_prob: float
            Log probability of selected action
        value: float
            State value estimate
        """
        if not valid_actions:
            return 0, 0.0, 0.0
        
        # Flatten state for neural network
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        
        # Get action probabilities and value
        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor)
        
        # Create mask for valid actions
        action_mask = torch.zeros(self.action_dim)
        for i, action in enumerate(valid_actions):
            # Map action tuple to action index (simplified mapping)
            action_idx = self._action_tuple_to_index(action)
            if action_idx < self.action_dim:
                action_mask[action_idx] = 1.0
        
        # Apply mask and normalize
        masked_probs = action_probs * action_mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # If no valid actions, use uniform distribution
            masked_probs = action_mask / action_mask.sum()
        
        # Add small epsilon to avoid NaN
        masked_probs = masked_probs + 1e-8
        masked_probs = masked_probs / masked_probs.sum()
        
        # Sample action
        dist = Categorical(masked_probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        # Convert back to valid action index
        valid_action_idx = self._get_valid_action_index(action_idx.item(), valid_actions)
        
        return valid_action_idx, log_prob.item(), value.item()
    
    def _action_tuple_to_index(self, action_tuple: Tuple[int, int]) -> int:
        """Convert action tuple (δ^o, δ^d) to action index"""
        delta_o, delta_d = action_tuple
        # Map from [-47, 47] x [-47, 47] to [0, 9024]
        # delta_o: -47 to 47 -> 0 to 94
        # delta_d: -47 to 47 -> 0 to 94
        # Combined: (delta_o + 47) * 95 + (delta_d + 47)
        return (delta_o + 47) * 95 + (delta_d + 47)
    
    def _get_valid_action_index(self, action_idx: int, valid_actions: List[Tuple[int, int]]) -> int:
        """Get index of action in valid_actions list"""
        action_tuple = self._action_index_to_tuple(action_idx)
        try:
            return valid_actions.index(action_tuple)
        except ValueError:
            return 0  # Default to first valid action
    
    def _action_index_to_tuple(self, action_idx: int) -> Tuple[int, int]:
        """Convert action index back to action tuple"""
        delta_o = (action_idx // 95) - 47
        delta_d = (action_idx % 95) - 47
        return (delta_o, delta_d)
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition in memory"""
        # Handle NaN rewards
        if np.isnan(reward) or np.isinf(reward):
            reward = -10.0  # Large penalty for invalid rewards
        
        self.memory.store(state, action, reward, next_state, done, log_prob, value)
    
    def update(self):
        """Update policy using PPO"""
        if len(self.memory.states) < 8:  # Minimum batch size
            return
        
        # Get data from memory
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(self.memory.next_states)).to(self.device)
        dones = torch.BoolTensor(self.memory.dones).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.memory.values).to(self.device)
        
        # Handle NaN values
        rewards = torch.nan_to_num(rewards, nan=-10.0, posinf=100.0, neginf=-100.0)
        old_values = torch.nan_to_num(old_values, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, dones)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy outputs
            action_probs, values = self.actor_critic(states)
            
            # Handle NaN in action probabilities
            action_probs = torch.nan_to_num(action_probs, nan=1e-8, posinf=1.0, neginf=1e-8)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            
            # Compute new log probabilities
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        self.memory.clear()
    
    def _compute_returns(self, rewards, dones):
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim: Tuple[int, ...], action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Calculate input size from flattened state
        input_size = np.prod(state_dim)
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_probs, value


class PPOMemory:
    """Experience buffer for PPO"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state.flatten())
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state.flatten() if next_state is not None else np.zeros_like(state.flatten()))
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


def train_ppo(env: DVRPEnvironment, 
              agent: PPOAgent, 
              num_episodes: int = 1000,
              max_steps_per_episode: int = 100,
              update_frequency: int = 32,
              save_frequency: int = 100,
              save_path: str = "models/ppo_dvrp.pth"):
    """
    Train PPO agent on DVRP environment
    
    Parameters:
    -----------
    env: DVRPEnvironment
        The DVRP environment
    agent: PPOAgent
        The PPO agent
    num_episodes: int
        Number of training episodes
    max_steps_per_episode: int
        Maximum steps per episode
    update_frequency: int
        Update policy every N steps
    save_frequency: int
        Save model every N episodes
    save_path: str
        Path to save the model
    """
    
    episode_rewards = []
    episode_costs = []
    
    for episode in range(num_episodes):
        state, request = env.reset()
        episode_reward = 0
        step = 0
        
        while step < max_steps_per_episode:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Get action from agent
            action_idx, log_prob, value = agent.get_action(state, valid_actions)
            action = valid_actions[action_idx]
            
            # Execute action
            next_state, next_request, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action_idx, reward, next_state, done, log_prob, value)
            
            # Update state
            state = next_state
            episode_reward += reward
            step += 1
            
            # Update agent
            if len(agent.memory.states) >= update_frequency:
                agent.update()
            
            if done:
                break
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        if env.route.cost and 'total_cost' in env.route.cost:
            episode_costs.append(env.route.cost['total_cost'])
        else:
            episode_costs.append(float('inf'))
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_cost = np.mean(episode_costs[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Cost: {avg_cost:.2f}")
        
        # Save model
        if episode % save_frequency == 0 and episode > 0:
            agent.save(save_path)
            print(f"Model saved at episode {episode}")
    
    return episode_rewards, episode_costs


# Example usage
if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    route = Route(N=20, auto_solve=False)
    env = DVRPEnvironment(route, acceptance_rate=0.7, alpha=2.0)
    
    # Get state dimensions
    state, _ = env.reset()
    state_dim = state.shape
    action_dim = 95 * 95  # 95 x 95 action space (δ^o, δ^d from -47 to 47)
    
    # Create PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        device=device
    )
    
    # Train agent
    print("Starting PPO training...")
    episode_rewards, episode_costs = train_ppo(
        env=env,
        agent=agent,
        num_episodes=1000,
        max_steps_per_episode=20,
        update_frequency=32,
        save_frequency=100,
        save_path="models/ppo_dvrp.pth"
    )
    
    print("Training completed!")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final average cost: {np.mean(episode_costs[-100:]):.2f}")
