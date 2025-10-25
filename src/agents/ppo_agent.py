"""
Proximal Policy Optimization (PPO) Agent for Portfolio Optimization

This module implements a PPO agent specifically designed for portfolio optimization
with continuous action spaces and portfolio weight constraints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple, Any
import os

from .base_agent import BaseAgent, PortfolioNetwork, softmax_weights, validate_portfolio_weights


class PPOPolicy(nn.Module):
    """
    PPO Policy Network for portfolio optimization.
    
    This network outputs both mean and log_std for continuous actions,
    which are then converted to portfolio weights using softmax.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 activation: str = "relu",
                 dropout_rate: float = 0.1):
        """
        Initialize the PPO policy network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_extractor = PortfolioNetwork(
            state_dim, action_dim, hidden_dims, activation, dropout_rate
        )
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in [self.mean_head, self.log_std_head]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        features = self.feature_extractor.network[:-1](state)  # All layers except last
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            state: Input state tensor
            deterministic: Whether to use deterministic action (mean)
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(mean)
        else:
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO loss calculation.
        
        Args:
            state: Input state tensor
            action: Action tensor to evaluate
            
        Returns:
            Tuple of (log_prob, entropy, value)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, torch.zeros_like(log_prob)  # No value function for now


class PPOAgent(BaseAgent):
    """
    PPO Agent for Portfolio Optimization.
    
    This agent uses PPO algorithm with continuous actions that are converted
    to portfolio weights using softmax normalization.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 batch_size: int = 64,
                 device: str = "auto"):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            batch_size: Batch size for training
            device: Device to run on
        """
        super().__init__(state_dim, action_dim, learning_rate, device)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Networks
        self.policy = PPOPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Training metrics
        self.episode_reward = 0
        self.episode_length = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Selected action (portfolio weights)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, log_prob = self.policy.get_action(state_tensor, deterministic=not training)
            action_logits = action_logits.squeeze(0).cpu().numpy()
            log_prob = log_prob.squeeze(0).cpu().numpy()
        
        # Convert to portfolio weights using softmax
        portfolio_weights = softmax_weights(action_logits)
        
        # Validate weights
        if not validate_portfolio_weights(portfolio_weights):
            print("Warning: Invalid portfolio weights generated, using equal weights")
            portfolio_weights = np.ones(self.action_dim) / self.action_dim
        
        # Store for training
        if training:
            self.states.append(state)
            self.actions.append(action_logits)  # Store raw logits for PPO
            self.log_probs.append(log_prob)
            self.values.append(0.0)  # Placeholder for value function
        
        return portfolio_weights
    
    def store_reward(self, reward: float, done: bool):
        """
        Store reward and done flag for the current step.
        
        Args:
            reward: Reward received
            done: Whether episode is done
        """
        self.rewards.append(reward)
        self.dones.append(done)
        
        self.episode_reward += reward
        self.episode_length += 1
        
        if done:
            self.training_metrics['episode_rewards'].append(self.episode_reward)
            self.training_metrics['episode_lengths'].append(self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0
    
    def update(self, batch: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Update the agent's parameters using stored experience.
        
        Args:
            batch: Not used for PPO (uses stored experience)
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.states) < self.batch_size:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        log_probs_old = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        dones = torch.BoolTensor(self.dones).to(self.device)
        
        # Calculate returns and advantages using GAE
        returns, advantages = self._calculate_gae(rewards, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0
        
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                
                # Forward pass
                log_probs, entropy_batch, values = self.policy.evaluate_actions(batch_states, batch_actions)
                
                # Calculate ratios
                ratios = torch.exp(log_probs - batch_log_probs_old)
                
                # Calculate surrogates
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss_batch = -torch.min(surr1, surr2).mean()
                
                # Value function loss (simplified)
                value_loss_batch = torch.nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy_batch.mean()
                
                # Total loss
                loss = policy_loss_batch + self.value_coef * value_loss_batch + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                policy_loss += policy_loss_batch.item()
                value_loss += value_loss_batch.item()
                entropy += entropy_batch.mean().item()
        
        # Clear stored experience
        self._clear_experience()
        
        # Store metrics
        metrics = {
            'loss': total_loss / (self.ppo_epochs * (len(states) // self.batch_size + 1)),
            'policy_loss': policy_loss / (self.ppo_epochs * (len(states) // self.batch_size + 1)),
            'value_loss': value_loss / (self.ppo_epochs * (len(states) // self.batch_size + 1)),
            'entropy': entropy / (self.ppo_epochs * (len(states) // self.batch_size + 1))
        }
        
        self.training_metrics['losses'].append(metrics['loss'])
        
        return metrics
    
    def _calculate_gae(self, rewards: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward tensor
            dones: Done flag tensor
            
        Returns:
            Tuple of (returns, advantages)
        """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Calculate returns
        running_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # Calculate advantages using GAE
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_advantage = 0
            delta = rewards[t] + self.gamma * (0 if t == len(rewards) - 1 else returns[t + 1]) - 0  # No value function
            running_advantage = delta + self.gamma * self.gae_lambda * running_advantage
            advantages[t] = running_advantage
        
        return returns, advantages
    
    def _clear_experience(self):
        """Clear stored experience."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save(self, filepath: str) -> None:
        """
        Save the agent's parameters to a file.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'training_metrics': self.training_metrics
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load the agent's parameters from a file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        
        print(f"Model loaded from {filepath}")
    
    def get_portfolio_weights(self, state: np.ndarray) -> np.ndarray:
        """
        Get portfolio weights for a given state (deterministic).
        
        Args:
            state: Current state observation
            
        Returns:
            Portfolio weights
        """
        return self.select_action(state, training=False)


if __name__ == "__main__":
    print("PPO Agent Module")
    print("=" * 50)
    print("This module implements PPO agent for portfolio optimization.")
    print("Use PPOAgent for continuous action spaces with portfolio constraints.")
