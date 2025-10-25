"""
Deep Q-Network (DQN) Agent for Portfolio Optimization

This module implements a DQN agent for discrete action spaces in portfolio optimization.
Actions represent discrete portfolio allocation decisions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, List, Optional, Tuple, Any
import os

from .base_agent import BaseAgent, PortfolioNetwork, ReplayBuffer, softmax_weights


class DQNAgent(BaseAgent):
    """
    DQN Agent for Portfolio Optimization with Discrete Actions.
    
    This agent uses discrete actions where each action represents a specific
    portfolio allocation strategy (e.g., equal weight, momentum, contrarian).
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 device: str = "auto"):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space (number of discrete actions)
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy
            epsilon_end: Final epsilon for epsilon-greedy
            epsilon_decay: Epsilon decay rate
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run on
        """
        super().__init__(state_dim, action_dim, learning_rate, device)
        
        # DQN hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = PortfolioNetwork(state_dim, action_dim).to(self.device)
        self.target_network = PortfolioNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, 1)  # Discrete actions
        
        # Training metrics
        self.step_count = 0
        self.episode_reward = 0
        self.episode_length = 0
        
        # Action mapping for portfolio strategies
        self.action_strategies = self._create_action_strategies()
    
    def _create_action_strategies(self) -> List[np.ndarray]:
        """
        Create discrete action strategies for portfolio allocation.
        
        Returns:
            List of portfolio weight vectors for each action
        """
        strategies = []
        
        # Action 0: Equal weight
        strategies.append(np.ones(self.action_dim) / self.action_dim)
        
        # Action 1: Momentum (favor recent winners)
        momentum_weights = np.zeros(self.action_dim)
        momentum_weights[0] = 0.4  # Assume first stock is momentum leader
        momentum_weights[1] = 0.3
        momentum_weights[2] = 0.2
        momentum_weights[3:] = 0.1 / (self.action_dim - 3)
        strategies.append(momentum_weights)
        
        # Action 2: Contrarian (favor recent losers)
        contrarian_weights = np.zeros(self.action_dim)
        contrarian_weights[-1] = 0.4  # Assume last stock is contrarian pick
        contrarian_weights[-2] = 0.3
        contrarian_weights[-3] = 0.2
        contrarian_weights[:-3] = 0.1 / (self.action_dim - 3)
        strategies.append(contrarian_weights)
        
        # Action 3: Conservative (favor low volatility)
        conservative_weights = np.zeros(self.action_dim)
        conservative_weights[0] = 0.5  # Assume first stock is conservative
        conservative_weights[1] = 0.3
        conservative_weights[2:] = 0.2 / (self.action_dim - 2)
        strategies.append(conservative_weights)
        
        # Action 4: Aggressive (favor high volatility)
        aggressive_weights = np.zeros(self.action_dim)
        aggressive_weights[-1] = 0.5  # Assume last stock is aggressive
        aggressive_weights[-2] = 0.3
        aggressive_weights[:-2] = 0.2 / (self.action_dim - 2)
        strategies.append(aggressive_weights)
        
        # Add more strategies if needed
        while len(strategies) < self.action_dim:
            # Random strategy
            random_weights = np.random.dirichlet(np.ones(self.action_dim))
            strategies.append(random_weights)
        
        return strategies[:self.action_dim]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Selected action (portfolio weights)
        """
        if training and random.random() < self.epsilon:
            # Random action
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        # Get portfolio weights for the selected action
        portfolio_weights = self.action_strategies[action_idx].copy()
        
        # Store for training
        if training:
            self.states.append(state)
            self.actions.append(action_idx)
            self.episode_reward += 0  # Will be updated when reward is received
            self.episode_length += 1
        
        return portfolio_weights
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """
        Store experience in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken (index)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.add(state, np.array([action]), reward, next_state, done)
        
        if done:
            self.training_metrics['episode_rewards'].append(self.episode_reward)
            self.training_metrics['episode_lengths'].append(self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0
    
    def update(self, batch: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Update the agent's parameters using experience from replay buffer.
        
        Args:
            batch: Not used for DQN (uses replay buffer)
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0, 'q_value': 0.0, 'epsilon': self.epsilon}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Store metrics
        metrics = {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon
        }
        
        self.training_metrics['losses'].append(metrics['loss'])
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the agent's parameters to a file.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'training_metrics': self.training_metrics,
            'action_strategies': self.action_strategies
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load the agent's parameters from a file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        self.action_strategies = checkpoint.get('action_strategies', self.action_strategies)
        
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
    
    def get_action_strategies(self) -> List[np.ndarray]:
        """
        Get the current action strategies.
        
        Returns:
            List of portfolio weight vectors for each action
        """
        return self.action_strategies.copy()


if __name__ == "__main__":
    print("DQN Agent Module")
    print("=" * 50)
    print("This module implements DQN agent for discrete action spaces.")
    print("Use DQNAgent for discrete portfolio allocation strategies.")
