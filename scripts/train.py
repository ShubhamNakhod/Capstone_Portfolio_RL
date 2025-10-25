"""
Training Script for Portfolio Optimization RL Agents

This script provides a comprehensive training pipeline for different RL agents
on the portfolio optimization environment.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import load_and_prepare_data
from environment.portfolio_env import create_portfolio_env
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent


def train_agent(env, agent, n_episodes: int = 1000, 
                save_freq: int = 100, save_dir: str = "models") -> Dict[str, List[float]]:
    """
    Train an RL agent on the portfolio environment.
    
    Args:
        env: Portfolio environment
        agent: RL agent to train
        n_episodes: Number of training episodes
        save_freq: Frequency of model saving
        save_dir: Directory to save models
        
    Returns:
        Dictionary of training metrics
    """
    print(f"Starting training for {n_episodes} episodes...")
    print(f"Environment: {env.__class__.__name__}")
    print(f"Agent: {agent.__class__.__name__}")
    print(f"State space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print("-" * 50)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    # Training loop
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience (for agents that need it)
            if hasattr(agent, 'store_reward'):
                agent.store_reward(reward, done or truncated)
            elif hasattr(agent, 'store_experience'):
                agent.store_experience(state, action, reward, next_state, done or truncated)
            
            # Update agent
            if episode_length % 10 == 0:  # Update every 10 steps
                update_metrics = agent.update()
                if update_metrics:
                    losses.append(update_metrics.get('loss', 0))
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if truncated:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-50:]) if len(episode_lengths) >= 50 else np.mean(episode_lengths)
            print(f"Episode {episode:4d} | Avg Reward: {avg_reward:8.4f} | Avg Length: {avg_length:6.1f}")
        
        # Save model
        if episode % save_freq == 0 and episode > 0:
            model_path = os.path.join(save_dir, f"model_episode_{episode}.pth")
            agent.save(model_path)
    
    # Save final model
    final_model_path = os.path.join(save_dir, "final_model.pth")
    agent.save(final_model_path)
    
    # Save training metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'n_episodes': n_episodes,
        'agent_type': agent.__class__.__name__,
        'env_type': env.__class__.__name__
    }
    
    metrics_path = os.path.join(save_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Models saved to: {save_dir}")
    
    return metrics


def plot_training_metrics(metrics: Dict[str, List[float]], save_dir: str = "results"):
    """
    Plot and save training metrics.
    
    Args:
        metrics: Training metrics dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Metrics - {metrics['agent_type']}", fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average of rewards
    window = min(50, len(metrics['episode_rewards']) // 10)
    if window > 1:
        moving_avg = pd.Series(metrics['episode_rewards']).rolling(window=window).mean()
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(metrics['episode_lengths'])
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Losses (if available)
    if metrics['losses']:
        axes[1, 1].plot(metrics['losses'])
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No loss data available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Loss')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f"training_metrics_{metrics['agent_type'].lower()}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training plots saved to: {plot_path}")


def evaluate_agent(env, agent, n_episodes: int = 100) -> Dict[str, float]:
    """
    Evaluate a trained agent on the environment.
    
    Args:
        env: Portfolio environment
        agent: Trained RL agent
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating agent for {n_episodes} episodes...")
    
    episode_rewards = []
    episode_returns = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action (no training)
            action = agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_returns.append(info['total_return'])
        episode_lengths.append(episode_length)
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'n_episodes': n_episodes
    }
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
    print(f"  Mean Return: {metrics['mean_return']:.4f} ± {metrics['std_return']:.4f}")
    print(f"  Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train RL agents for portfolio optimization')
    parser.add_argument('--agent', type=str, default='ppo', choices=['ppo', 'dqn'],
                       help='Agent type to train')
    parser.add_argument('--env', type=str, default='enhanced', choices=['enhanced', 'expanded'],
                       help='Environment type')
    parser.add_argument('--stocks', type=str, default='diversified', 
                       choices=['diversified', 'simple', 'all'],
                       help='Stock selection type')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--save_freq', type=int, default=100,
                       help='Frequency of model saving')
    parser.add_argument('--data_dir', type=str, default='../Files',
                       help='Directory containing stock data')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("Portfolio Optimization RL Training")
    print("=" * 50)
    print(f"Agent: {args.agent}")
    print(f"Environment: {args.env}")
    print(f"Stocks: {args.stocks}")
    print(f"Episodes: {args.episodes}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    try:
        prices_df, returns_df, quality_metrics = load_and_prepare_data(
            args.data_dir, stock_selection=args.stocks
        )
        print(f"Data loaded successfully: {prices_df.shape[0]} days, {prices_df.shape[1]} stocks")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create environment
    print("Creating environment...")
    try:
        env = create_portfolio_env(
            env_type=args.env,
            prices_df=prices_df,
            returns_df=returns_df,
            episode_length=252,  # 1 trading year
            lookback_window=20
        )
        print(f"Environment created: {env.__class__.__name__}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    # Create agent
    print("Creating agent...")
    try:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
        
        if args.agent == 'ppo':
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=3e-4,
                gamma=0.99,
                clip_ratio=0.2,
                ppo_epochs=4,
                batch_size=64
            )
        elif args.agent == 'dqn':
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=5,  # 5 discrete strategies
                learning_rate=1e-4,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995
            )
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")
        
        print(f"Agent created: {agent.__class__.__name__}")
    except Exception as e:
        print(f"Error creating agent: {e}")
        return
    
    # Train agent
    print("Starting training...")
    start_time = time.time()
    
    try:
        metrics = train_agent(
            env=env,
            agent=agent,
            n_episodes=args.episodes,
            save_freq=args.save_freq,
            save_dir=args.save_dir
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Plot metrics
    print("Plotting training metrics...")
    try:
        plot_training_metrics(metrics, args.results_dir)
    except Exception as e:
        print(f"Error plotting metrics: {e}")
    
    # Evaluate agent
    print("Evaluating trained agent...")
    try:
        eval_metrics = evaluate_agent(env, agent, n_episodes=100)
        
        # Save evaluation results
        eval_path = os.path.join(args.results_dir, f"evaluation_{args.agent}.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"Evaluation results saved to: {eval_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    print("\nTraining pipeline completed!")


if __name__ == "__main__":
    main()
