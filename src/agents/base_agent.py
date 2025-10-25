"""
Base Agent Class for Portfolio Optimization

This module provides the base class for all reinforcement learning agents
used in portfolio optimization.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym


class BaseAgent(ABC):
    """
    Abstract base class for all portfolio optimization agents.
    
    This class defines the interface that all agents must implement
    and provides common functionality.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-4,
                 device: str = "auto"):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            learning_rate: Learning rate for optimization
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Agent initialized on device: {self.device}")
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'learning_rate': learning_rate
        }
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update the agent's parameters using a batch of experience.
        
        Args:
            batch: Dictionary containing experience batch
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the agent's parameters to a file.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load the agent's parameters from a file.
        
        Args:
            filepath: Path to load the model from
        """
        pass
    
    def get_training_metrics(self) -> Dict[str, List[float]]:
        """
        Get training metrics.
        
        Returns:
            Dictionary of training metrics
        """
        return self.training_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset training metrics."""
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'learning_rate': self.learning_rate
        }
    
    def update_learning_rate(self, new_lr: float) -> None:
        """
        Update the learning rate.
        
        Args:
            new_lr: New learning rate
        """
        self.learning_rate = new_lr
        self.training_metrics['learning_rate'] = new_lr


class PortfolioNetwork(nn.Module):
    """
    Neural network for portfolio optimization.
    
    This network takes market state as input and outputs portfolio weights.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 activation: str = "relu",
                 dropout_rate: float = 0.1):
        """
        Initialize the portfolio network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("relu", "tanh", "gelu")
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1)
        }
        return activations.get(activation, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Output action tensor
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling experience.
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 state_dim: int = None,
                 action_dim: int = None):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        
        self.size = 0
        self.ptr = 0
    
    def add(self, 
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """
        Add experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary containing the sampled batch
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
    
    def __len__(self):
        """Return current size of the buffer."""
        return self.size


def softmax_weights(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert logits to portfolio weights using softmax.
    
    Args:
        logits: Raw action logits
        temperature: Temperature parameter for softmax
        
    Returns:
        Normalized portfolio weights
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Numerical stability
    scaled_logits = scaled_logits - np.max(scaled_logits)
    
    # Softmax
    exp_logits = np.exp(scaled_logits)
    weights = exp_logits / np.sum(exp_logits)
    
    return weights


def validate_portfolio_weights(weights: np.ndarray, 
                              tolerance: float = 1e-6) -> bool:
    """
    Validate that portfolio weights sum to 1 and are non-negative.
    
    Args:
        weights: Portfolio weights to validate
        tolerance: Tolerance for sum validation
        
    Returns:
        True if weights are valid, False otherwise
    """
    # Check non-negative
    if np.any(weights < 0):
        return False
    
    # Check sum to 1
    if abs(np.sum(weights) - 1.0) > tolerance:
        return False
    
    return True


if __name__ == "__main__":
    print("Base Agent Module")
    print("=" * 50)
    print("This module provides the base class and utilities for RL agents.")
    print("Use BaseAgent as the parent class for all agent implementations.")
