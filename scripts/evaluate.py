"""
Evaluation Script for Portfolio Optimization RL Agents

This script provides comprehensive evaluation of trained RL agents including
backtesting, performance comparison, and visualization.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import load_and_prepare_data, prepare_training_data
from environment.portfolio_env import create_portfolio_env
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent


def run_backtest(env, agent, n_episodes: int = 100, 
                verbose: bool = False) -> Dict[str, Any]:
    """
    Run backtesting on the trained agent.
    
    Args:
        env: Portfolio environment
        agent: Trained RL agent
        n_episodes: Number of backtest episodes
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary of backtest results
    """
    print(f"Running backtest for {n_episodes} episodes...")
    
    episode_results = []
    portfolio_histories = []
    weight_histories = []
    
    for episode in range(n_episodes):
        if verbose and episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes}")
        
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        portfolio_values = [env.initial_capital]
        portfolio_weights = []
        done = False
        
        while not done:
            # Select action (no training)
            action = agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store portfolio state
            portfolio_values.append(info['portfolio_value'])
            portfolio_weights.append(info['portfolio_weights'].copy())
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if truncated:
                break
        
        # Store episode results
        episode_results.append({
            'episode': episode,
            'total_return': info['total_return'],
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'final_value': info['portfolio_value'],
            'max_drawdown': calculate_max_drawdown(portfolio_values),
            'volatility': calculate_volatility(portfolio_values),
            'sharpe_ratio': calculate_sharpe_ratio(portfolio_values)
        })
        
        portfolio_histories.append(portfolio_values)
        weight_histories.append(portfolio_weights)
    
    # Calculate summary statistics
    returns = [result['total_return'] for result in episode_results]
    rewards = [result['episode_reward'] for result in episode_results]
    lengths = [result['episode_length'] for result in episode_results]
    drawdowns = [result['max_drawdown'] for result in episode_results]
    volatilities = [result['volatility'] for result in episode_results]
    sharpe_ratios = [result['sharpe_ratio'] for result in episode_results]
    
    summary = {
        'n_episodes': n_episodes,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'mean_drawdown': np.mean(drawdowns),
        'std_drawdown': np.std(drawdowns),
        'mean_volatility': np.mean(volatilities),
        'std_volatility': np.std(volatilities),
        'mean_sharpe': np.mean(sharpe_ratios),
        'std_sharpe': np.std(sharpe_ratios),
        'win_rate': np.mean([r > 0 for r in returns]),
        'episode_results': episode_results,
        'portfolio_histories': portfolio_histories,
        'weight_histories': weight_histories
    }
    
    print(f"Backtest completed!")
    print(f"Mean Return: {summary['mean_return']:.4f} ± {summary['std_return']:.4f}")
    print(f"Mean Sharpe: {summary['mean_sharpe']:.4f} ± {summary['std_sharpe']:.4f}")
    print(f"Win Rate: {summary['win_rate']:.2%}")
    print(f"Mean Drawdown: {summary['mean_drawdown']:.4f} ± {summary['std_drawdown']:.4f}")
    
    return summary


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate maximum drawdown from portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0
    
    peak = portfolio_values[0]
    max_dd = 0.0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    
    return max_dd


def calculate_volatility(portfolio_values: List[float]) -> float:
    """Calculate volatility from portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return np.std(returns) * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(portfolio_values: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from portfolio values."""
    if len(portfolio_values) < 2:
        return 0.0
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_return = np.mean(returns) * 252 - risk_free_rate
    volatility = np.std(returns) * np.sqrt(252)
    
    return excess_return / volatility


def compare_with_benchmarks(env, agent, n_episodes: int = 100) -> Dict[str, Any]:
    """
    Compare agent performance with benchmark strategies.
    
    Args:
        env: Portfolio environment
        agent: Trained RL agent
        n_episodes: Number of comparison episodes
        
    Returns:
        Dictionary of comparison results
    """
    print("Comparing with benchmark strategies...")
    
    # Run agent backtest
    agent_results = run_backtest(env, agent, n_episodes, verbose=False)
    
    # Run benchmark strategies
    benchmarks = {}
    
    # Equal weight benchmark
    benchmarks['equal_weight'] = run_equal_weight_benchmark(env, n_episodes)
    
    # Buy and hold benchmark (first stock)
    benchmarks['buy_hold'] = run_buy_hold_benchmark(env, n_episodes)
    
    # Random benchmark
    benchmarks['random'] = run_random_benchmark(env, n_episodes)
    
    # Compare results
    comparison = {
        'agent': {
            'mean_return': agent_results['mean_return'],
            'std_return': agent_results['std_return'],
            'mean_sharpe': agent_results['mean_sharpe'],
            'mean_drawdown': agent_results['mean_drawdown'],
            'win_rate': agent_results['win_rate']
        },
        'benchmarks': benchmarks,
        'n_episodes': n_episodes
    }
    
    print("\nPerformance Comparison:")
    print(f"{'Strategy':<15} {'Mean Return':<12} {'Std Return':<12} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<8}")
    print("-" * 80)
    
    for name, results in comparison['benchmarks'].items():
        print(f"{name:<15} {results['mean_return']:<12.4f} {results['std_return']:<12.4f} "
              f"{results['mean_sharpe']:<8.4f} {results['mean_drawdown']:<10.4f} {results['win_rate']:<8.2%}")
    
    print(f"{'RL Agent':<15} {comparison['agent']['mean_return']:<12.4f} {comparison['agent']['std_return']:<12.4f} "
          f"{comparison['agent']['mean_sharpe']:<8.4f} {comparison['agent']['mean_drawdown']:<10.4f} {comparison['agent']['win_rate']:<8.2%}")
    
    return comparison


def run_equal_weight_benchmark(env, n_episodes: int) -> Dict[str, float]:
    """Run equal weight benchmark strategy."""
    episode_returns = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            # Equal weight action
            action = np.ones(env.action_space.shape[0]) / env.action_space.shape[0]
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            
            if truncated:
                break
        
        episode_returns.append(info['total_return'])
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_sharpe': 0.0,  # Simplified
        'mean_drawdown': 0.0,  # Simplified
        'win_rate': np.mean([r > 0 for r in episode_returns])
    }


def run_buy_hold_benchmark(env, n_episodes: int) -> Dict[str, float]:
    """Run buy and hold benchmark strategy."""
    episode_returns = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            # Buy and hold first stock
            action = np.zeros(env.action_space.shape[0])
            action[0] = 1.0
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            
            if truncated:
                break
        
        episode_returns.append(info['total_return'])
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_sharpe': 0.0,  # Simplified
        'mean_drawdown': 0.0,  # Simplified
        'win_rate': np.mean([r > 0 for r in episode_returns])
    }


def run_random_benchmark(env, n_episodes: int) -> Dict[str, float]:
    """Run random benchmark strategy."""
    episode_returns = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            # Random action
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            
            if truncated:
                break
        
        episode_returns.append(info['total_return'])
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_sharpe': 0.0,  # Simplified
        'mean_drawdown': 0.0,  # Simplified
        'win_rate': np.mean([r > 0 for r in episode_returns])
    }


def plot_backtest_results(backtest_results: Dict[str, Any], save_dir: str = "results"):
    """
    Plot backtest results.
    
    Args:
        backtest_results: Results from run_backtest
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    returns = [result['total_return'] for result in backtest_results['episode_results']]
    rewards = [result['episode_reward'] for result in backtest_results['episode_results']]
    drawdowns = [result['max_drawdown'] for result in backtest_results['episode_results']]
    sharpe_ratios = [result['sharpe_ratio'] for result in backtest_results['episode_results']]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backtest Results', fontsize=16)
    
    # Returns distribution
    axes[0, 0].hist(returns, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.4f}')
    axes[0, 0].set_title('Returns Distribution')
    axes[0, 0].set_xlabel('Total Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rewards over episodes
    axes[0, 1].plot(rewards)
    axes[0, 1].set_title('Episode Rewards')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Drawdowns
    axes[1, 0].hist(drawdowns, bins=30, alpha=0.7, edgecolor='black', color='red')
    axes[1, 0].axvline(np.mean(drawdowns), color='darkred', linestyle='--', label=f'Mean: {np.mean(drawdowns):.4f}')
    axes[1, 0].set_title('Maximum Drawdowns')
    axes[1, 0].set_xlabel('Max Drawdown')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sharpe ratios
    axes[1, 1].hist(sharpe_ratios, bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[1, 1].axvline(np.mean(sharpe_ratios), color='darkgreen', linestyle='--', label=f'Mean: {np.mean(sharpe_ratios):.4f}')
    axes[1, 1].set_title('Sharpe Ratios')
    axes[1, 1].set_xlabel('Sharpe Ratio')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, "backtest_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Backtest plots saved to: {plot_path}")


def plot_comparison_results(comparison_results: Dict[str, Any], save_dir: str = "results"):
    """
    Plot comparison results with benchmarks.
    
    Args:
        comparison_results: Results from compare_with_benchmarks
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    strategies = list(comparison_results['benchmarks'].keys()) + ['RL Agent']
    mean_returns = [comparison_results['benchmarks'][s]['mean_return'] for s in comparison_results['benchmarks']] + [comparison_results['agent']['mean_return']]
    std_returns = [comparison_results['benchmarks'][s]['std_return'] for s in comparison_results['benchmarks']] + [comparison_results['agent']['std_return']]
    sharpe_ratios = [comparison_results['benchmarks'][s]['mean_sharpe'] for s in comparison_results['benchmarks']] + [comparison_results['agent']['mean_sharpe']]
    win_rates = [comparison_results['benchmarks'][s]['win_rate'] for s in comparison_results['benchmarks']] + [comparison_results['agent']['win_rate']]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Strategy Comparison', fontsize=16)
    
    # Mean returns
    bars1 = axes[0, 0].bar(strategies, mean_returns, alpha=0.7)
    axes[0, 0].set_title('Mean Returns')
    axes[0, 0].set_ylabel('Mean Return')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add error bars
    axes[0, 0].errorbar(strategies, mean_returns, yerr=std_returns, fmt='none', color='black', capsize=5)
    
    # Sharpe ratios
    bars2 = axes[0, 1].bar(strategies, sharpe_ratios, alpha=0.7, color='green')
    axes[0, 1].set_title('Sharpe Ratios')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Win rates
    bars3 = axes[1, 0].bar(strategies, win_rates, alpha=0.7, color='orange')
    axes[1, 0].set_title('Win Rates')
    axes[1, 0].set_ylabel('Win Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Risk-return scatter
    axes[1, 1].scatter(std_returns, mean_returns, s=100, alpha=0.7)
    for i, strategy in enumerate(strategies):
        axes[1, 1].annotate(strategy, (std_returns[i], mean_returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 1].set_title('Risk vs Return')
    axes[1, 1].set_xlabel('Standard Deviation')
    axes[1, 1].set_ylabel('Mean Return')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, "strategy_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plots saved to: {plot_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate RL agents for portfolio optimization')
    parser.add_argument('--agent', type=str, default='ppo', choices=['ppo', 'dqn'],
                       help='Agent type to evaluate')
    parser.add_argument('--env', type=str, default='enhanced', choices=['enhanced', 'expanded'],
                       help='Environment type')
    parser.add_argument('--stocks', type=str, default='diversified', 
                       choices=['diversified', 'simple', 'all'],
                       help='Stock selection type')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--data_dir', type=str, default='../Files',
                       help='Directory containing stock data')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with benchmark strategies')
    
    args = parser.parse_args()
    
    print("Portfolio Optimization RL Evaluation")
    print("=" * 50)
    print(f"Agent: {args.agent}")
    print(f"Environment: {args.env}")
    print(f"Stocks: {args.stocks}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
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
    
    # Load trained model
    print("Loading trained model...")
    try:
        agent.load(args.model_path)
        print(f"Model loaded from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run backtest
    print("Running backtest...")
    try:
        backtest_results = run_backtest(env, agent, args.episodes, verbose=True)
        
        # Save backtest results
        backtest_path = os.path.join(args.results_dir, f"backtest_{args.agent}.json")
        with open(backtest_path, 'w') as f:
            json.dump(backtest_results, f, indent=2)
        print(f"Backtest results saved to: {backtest_path}")
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        return
    
    # Plot backtest results
    print("Plotting backtest results...")
    try:
        plot_backtest_results(backtest_results, args.results_dir)
    except Exception as e:
        print(f"Error plotting backtest results: {e}")
    
    # Compare with benchmarks
    if args.compare:
        print("Comparing with benchmark strategies...")
        try:
            comparison_results = compare_with_benchmarks(env, agent, args.episodes)
            
            # Save comparison results
            comparison_path = os.path.join(args.results_dir, f"comparison_{args.agent}.json")
            with open(comparison_path, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"Comparison results saved to: {comparison_path}")
            
            # Plot comparison results
            plot_comparison_results(comparison_results, args.results_dir)
            
        except Exception as e:
            print(f"Error during comparison: {e}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
