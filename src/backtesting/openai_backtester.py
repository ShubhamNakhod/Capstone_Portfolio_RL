"""
OpenAI Trading Bot Backtesting Framework

This module provides comprehensive backtesting capabilities for OpenAI-powered
trading bots, allowing you to evaluate performance on historical data and
compare against traditional strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Callable
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..agents.openai_trading_bot import OpenAITradingBot
from ..environment.openai_portfolio_env import OpenAIPortfolioEnv
from ..data.openai_data_integration import OpenAIDataIntegrator


class OpenAIBacktester:
    """
    Comprehensive backtesting framework for OpenAI trading bots.
    
    This class provides tools to backtest OpenAI trading strategies,
    compare performance against benchmarks, and generate detailed reports.
    """
    
    def __init__(self, 
                 api_key: str,
                 data_directory: str = "../Files",
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001):
        """
        Initialize the backtester.
        
        Args:
            api_key: OpenAI API key
            data_directory: Path to stock data files
            initial_capital: Starting capital
            transaction_cost: Transaction cost percentage
        """
        self.api_key = api_key
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Initialize data integrator
        self.data_integrator = OpenAIDataIntegrator(data_directory)
        
        # Load and prepare data
        self._prepare_data()
        
        # Results storage
        self.backtest_results = {}
        self.performance_metrics = {}
        
    def _prepare_data(self):
        """Prepare market data for backtesting."""
        print("Loading and preparing market data...")
        
        # Load stock data
        self.data_integrator.load_all_stock_data()
        
        # Create combined dataset
        self.combined_prices = self.data_integrator.create_combined_dataset()
        
        # Calculate returns
        self.daily_returns = self.data_integrator.calculate_returns()
        
        # Calculate technical indicators
        self.technical_indicators = self.data_integrator.calculate_technical_indicators()
        
        print(f"Data prepared: {len(self.combined_prices)} days, {len(self.combined_prices.columns)} stocks")
    
    def run_backtest(self, 
                    start_date: str = None,
                    end_date: str = None,
                    rebalance_frequency: str = 'daily',
                    model_name: str = 'gpt-4',
                    max_trades_per_day: int = 5,
                    verbose: bool = True) -> Dict:
        """
        Run backtest for OpenAI trading bot.
        
        Args:
            start_date: Start date for backtest (format: 'YYYY-MM-DD')
            end_date: End date for backtest
            rebalance_frequency: 'daily', 'weekly', or 'monthly'
            model_name: OpenAI model to use
            max_trades_per_day: Maximum number of trades per day
            verbose: Whether to print progress
            
        Returns:
            Dictionary with backtest results
        """
        # Set date range
        if start_date is None:
            start_date = self.combined_prices.index[100].strftime('%Y-%m-%d')  # Skip first 100 days
        if end_date is None:
            end_date = self.combined_prices.index[-1].strftime('%Y-%m-%d')
        
        start_idx = self.combined_prices.index.get_loc(pd.to_datetime(start_date), method='nearest')
        end_idx = self.combined_prices.index.get_loc(pd.to_datetime(end_date), method='nearest')
        
        # Create trading bot
        bot = OpenAITradingBot(self.api_key, self.initial_capital)
        
        # Determine rebalancing schedule
        if rebalance_frequency == 'daily':
            rebalance_dates = self.combined_prices.index[start_idx:end_idx+1]
        elif rebalance_frequency == 'weekly':
            rebalance_dates = self.combined_prices.index[start_idx:end_idx+1][::5]  # Every 5 days
        elif rebalance_frequency == 'monthly':
            rebalance_dates = self.combined_prices.index[start_idx:end_idx+1][::20]  # Every 20 days
        else:
            raise ValueError("Rebalance frequency must be 'daily', 'weekly', or 'monthly'")
        
        # Initialize tracking
        portfolio_values = [self.initial_capital]
        trade_history = []
        daily_returns = []
        
        if verbose:
            print(f"Running backtest from {start_date} to {end_date}")
            print(f"Rebalancing frequency: {rebalance_frequency}")
            print(f"Total rebalancing dates: {len(rebalance_dates)}")
        
        # Run backtest
        for i, date in enumerate(rebalance_dates):
            if verbose and i % 10 == 0:
                print(f"Processing date {i+1}/{len(rebalance_dates)}: {date.strftime('%Y-%m-%d')}")
            
            try:
                # Get market data for this date
                market_data = self.combined_prices.loc[:date].tail(50)  # Last 50 days
                current_prices = self.combined_prices.loc[date].to_dict()
                
                # Get OpenAI recommendation
                recommendation = bot.get_trading_recommendation(market_data)
                
                # Execute trade
                success = bot.execute_trade(recommendation, current_prices)
                
                # Update portfolio value
                portfolio_value = bot.update_portfolio_value(current_prices)
                
                # Store results
                portfolio_values.append(portfolio_value)
                
                if bot.trade_history:
                    trade_history.append({
                        'date': date,
                        'trade': bot.trade_history[-1],
                        'portfolio_value': portfolio_value
                    })
                
                # Calculate daily return
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_values[-1] / portfolio_values[-2]) - 1
                    daily_returns.append(daily_return)
                
                # Limit trades per day
                if len(bot.trade_history) > 0:
                    today_trades = [t for t in bot.trade_history if t['timestamp'].date() == date.date()]
                    if len(today_trades) >= max_trades_per_day:
                        if verbose:
                            print(f"  Max trades reached for {date.strftime('%Y-%m-%d')}")
                
            except Exception as e:
                if verbose:
                    print(f"Error on {date.strftime('%Y-%m-%d')}: {e}")
                continue
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_values, daily_returns, trade_history
        )
        
        # Store results
        backtest_results = {
            'start_date': start_date,
            'end_date': end_date,
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'trade_history': trade_history,
            'performance_metrics': performance_metrics,
            'final_portfolio': bot.get_portfolio_summary(),
            'rebalance_frequency': rebalance_frequency,
            'model_name': model_name
        }
        
        self.backtest_results['openai_bot'] = backtest_results
        
        if verbose:
            self._print_backtest_summary(performance_metrics)
        
        return backtest_results
    
    def _calculate_performance_metrics(self, 
                                     portfolio_values: List[float],
                                     daily_returns: List[float],
                                     trade_history: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(portfolio_values) < 2:
            return {}
        
        # Basic metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        
        # Volatility and Sharpe ratio
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized
            mean_return = np.mean(daily_returns) * 252  # Annualized
            sharpe_ratio = mean_return / (volatility / 100) if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Trade statistics
        total_trades = len(trade_history)
        buy_trades = sum(1 for trade in trade_history if trade['trade']['action'] == 'buy')
        sell_trades = sum(1 for trade in trade_history if trade['trade']['action'] == 'sell')
        
        # Win rate (simplified)
        if len(daily_returns) > 0:
            winning_days = sum(1 for ret in daily_returns if ret > 0)
            win_rate = (winning_days / len(daily_returns)) * 100
        else:
            win_rate = 0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'final_value': portfolio_values[-1],
            'initial_value': portfolio_values[0]
        }
    
    def _print_backtest_summary(self, metrics: Dict):
        """Print backtest summary."""
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Volatility: {metrics['volatility']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Final Value: ${metrics['final_value']:.2f}")
        print("="*50)
    
    def run_benchmark_comparison(self, 
                                start_date: str = None,
                                end_date: str = None) -> Dict:
        """
        Run benchmark strategies for comparison.
        
        Args:
            start_date: Start date for comparison
            end_date: End date for comparison
            
        Returns:
            Dictionary with benchmark results
        """
        if start_date is None:
            start_date = self.combined_prices.index[100].strftime('%Y-%m-%d')
        if end_date is None:
            end_date = self.combined_prices.index[-1].strftime('%Y-%m-%d')
        
        start_idx = self.combined_prices.index.get_loc(pd.to_datetime(start_date), method='nearest')
        end_idx = self.combined_prices.index.get_loc(pd.to_datetime(end_date), method='nearest')
        
        benchmark_data = self.combined_prices.iloc[start_idx:end_idx+1]
        
        # 1. Buy and Hold (Equal Weight)
        equal_weights = np.ones(len(benchmark_data.columns)) / len(benchmark_data.columns)
        bh_values = []
        for _, row in benchmark_data.iterrows():
            portfolio_value = np.sum(equal_weights * row.values) * self.initial_capital
            bh_values.append(portfolio_value)
        
        bh_returns = pd.Series(bh_values).pct_change().dropna().tolist()
        bh_metrics = self._calculate_performance_metrics(bh_values, bh_returns, [])
        
        # 2. Buy and Hold (Best Stock)
        best_stock = benchmark_data.iloc[-1] / benchmark_data.iloc[0]
        best_stock_name = best_stock.idxmax()
        best_stock_values = (benchmark_data[best_stock_name] / benchmark_data[best_stock_name].iloc[0] * self.initial_capital).tolist()
        best_stock_returns = pd.Series(best_stock_values).pct_change().dropna().tolist()
        best_stock_metrics = self._calculate_performance_metrics(best_stock_values, best_stock_returns, [])
        
        # 3. Buy and Hold (SP500 if available)
        sp500_metrics = {}
        if 'SP500' in benchmark_data.columns:
            sp500_values = (benchmark_data['SP500'] / benchmark_data['SP500'].iloc[0] * self.initial_capital).tolist()
            sp500_returns = pd.Series(sp500_values).pct_change().dropna().tolist()
            sp500_metrics = self._calculate_performance_metrics(sp500_values, sp500_returns, [])
        
        # Store benchmark results
        self.backtest_results['benchmarks'] = {
            'buy_hold_equal': {
                'name': 'Buy & Hold (Equal Weight)',
                'values': bh_values,
                'metrics': bh_metrics
            },
            'buy_hold_best': {
                'name': f'Buy & Hold ({best_stock_name})',
                'values': best_stock_values,
                'metrics': best_stock_metrics,
                'stock': best_stock_name
            },
            'buy_hold_sp500': {
                'name': 'Buy & Hold (SP500)',
                'values': sp500_values if 'SP500' in benchmark_data.columns else [],
                'metrics': sp500_metrics
            }
        }
        
        print("\nBenchmark strategies completed:")
        print(f"1. Buy & Hold (Equal Weight): {bh_metrics['total_return']:.2f}% return")
        print(f"2. Buy & Hold ({best_stock_name}): {best_stock_metrics['total_return']:.2f}% return")
        if sp500_metrics:
            print(f"3. Buy & Hold (SP500): {sp500_metrics['total_return']:.2f}% return")
        
        return self.backtest_results['benchmarks']
    
    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report.
        
        Returns:
            Formatted performance report string
        """
        if not self.backtest_results:
            return "No backtest results available. Run backtest first."
        
        report = []
        report.append("="*80)
        report.append("OPENAI TRADING BOT PERFORMANCE REPORT")
        report.append("="*80)
        
        # OpenAI Bot Results
        if 'openai_bot' in self.backtest_results:
            bot_results = self.backtest_results['openai_bot']
            metrics = bot_results['performance_metrics']
            
            report.append(f"\n🤖 OPENAI TRADING BOT RESULTS")
            report.append("-" * 40)
            report.append(f"Period: {bot_results['start_date']} to {bot_results['end_date']}")
            report.append(f"Model: {bot_results['model_name']}")
            report.append(f"Rebalancing: {bot_results['rebalance_frequency']}")
            report.append(f"Initial Capital: ${self.initial_capital:,.2f}")
            report.append(f"Final Value: ${metrics['final_value']:,.2f}")
            report.append(f"Total Return: {metrics['total_return']:.2f}%")
            report.append(f"Volatility: {metrics['volatility']:.2f}%")
            report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            report.append(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            report.append(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
            report.append(f"Win Rate: {metrics['win_rate']:.1f}%")
            report.append(f"Total Trades: {metrics['total_trades']}")
        
        # Benchmark Comparison
        if 'benchmarks' in self.backtest_results:
            report.append(f"\n📊 BENCHMARK COMPARISON")
            report.append("-" * 40)
            
            benchmarks = self.backtest_results['benchmarks']
            for key, benchmark in benchmarks.items():
                if benchmark['metrics']:
                    name = benchmark['name']
                    metrics = benchmark['metrics']
                    report.append(f"{name}:")
                    report.append(f"  Total Return: {metrics['total_return']:.2f}%")
                    report.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                    report.append(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Performance Comparison
        if 'openai_bot' in self.backtest_results and 'benchmarks' in self.backtest_results:
            report.append(f"\n🏆 PERFORMANCE RANKING")
            report.append("-" * 40)
            
            # Collect all strategies
            strategies = [('OpenAI Bot', self.backtest_results['openai_bot']['performance_metrics'])]
            
            for key, benchmark in self.backtest_results['benchmarks'].items():
                if benchmark['metrics']:
                    strategies.append((benchmark['name'], benchmark['metrics']))
            
            # Sort by total return
            strategies.sort(key=lambda x: x[1]['total_return'], reverse=True)
            
            for i, (name, metrics) in enumerate(strategies, 1):
                report.append(f"{i}. {name}: {metrics['total_return']:.2f}%")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if 'openai_bot' in self.backtest_results:
            bot_metrics = self.backtest_results['openai_bot']['performance_metrics']
            
            if bot_metrics['total_return'] > 0:
                report.append("✓ OpenAI bot generated positive returns")
            else:
                report.append("⚠ OpenAI bot generated negative returns")
            
            if bot_metrics['sharpe_ratio'] > 1.0:
                report.append("✓ Good risk-adjusted returns (Sharpe > 1.0)")
            elif bot_metrics['sharpe_ratio'] > 0.5:
                report.append("⚠ Moderate risk-adjusted returns (Sharpe > 0.5)")
            else:
                report.append("⚠ Poor risk-adjusted returns (Sharpe < 0.5)")
            
            if bot_metrics['max_drawdown'] > -20:
                report.append("✓ Acceptable maximum drawdown (< 20%)")
            else:
                report.append("⚠ High maximum drawdown (> 20%)")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def plot_performance(self, save_path: str = None):
        """
        Plot performance comparison charts.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.backtest_results:
            print("No backtest results to plot. Run backtest first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OpenAI Trading Bot Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Portfolio Value Over Time
        ax1 = axes[0, 0]
        
        if 'openai_bot' in self.backtest_results:
            bot_values = self.backtest_results['openai_bot']['portfolio_values']
            ax1.plot(bot_values, label='OpenAI Bot', linewidth=2, color='blue')
        
        if 'benchmarks' in self.backtest_results:
            for key, benchmark in self.backtest_results['benchmarks'].items():
                if benchmark['values']:
                    ax1.plot(benchmark['values'], label=benchmark['name'], alpha=0.7)
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Returns Distribution
        ax2 = axes[0, 1]
        
        if 'openai_bot' in self.backtest_results:
            bot_returns = self.backtest_results['openai_bot']['daily_returns']
            ax2.hist(bot_returns, bins=30, alpha=0.7, label='OpenAI Bot', color='blue')
        
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance Metrics Comparison
        ax3 = axes[1, 0]
        
        metrics_data = []
        labels = []
        
        if 'openai_bot' in self.backtest_results:
            bot_metrics = self.backtest_results['openai_bot']['performance_metrics']
            metrics_data.append(bot_metrics['total_return'])
            labels.append('OpenAI Bot')
        
        if 'benchmarks' in self.backtest_results:
            for key, benchmark in self.backtest_results['benchmarks'].items():
                if benchmark['metrics']:
                    metrics_data.append(benchmark['metrics']['total_return'])
                    labels.append(benchmark['name'])
        
        bars = ax3.bar(labels, metrics_data, alpha=0.7)
        ax3.set_title('Total Return Comparison')
        ax3.set_ylabel('Total Return (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Risk-Return Scatter
        ax4 = axes[1, 1]
        
        returns = []
        volatilities = []
        strategy_names = []
        
        if 'openai_bot' in self.backtest_results:
            bot_metrics = self.backtest_results['openai_bot']['performance_metrics']
            returns.append(bot_metrics['total_return'])
            volatilities.append(bot_metrics['volatility'])
            strategy_names.append('OpenAI Bot')
        
        if 'benchmarks' in self.backtest_results:
            for key, benchmark in self.backtest_results['benchmarks'].items():
                if benchmark['metrics']:
                    returns.append(benchmark['metrics']['total_return'])
                    volatilities.append(benchmark['metrics']['volatility'])
                    strategy_names.append(benchmark['name'])
        
        scatter = ax4.scatter(volatilities, returns, s=100, alpha=0.7, c=range(len(returns)), cmap='viridis')
        
        # Add labels
        for i, name in enumerate(strategy_names):
            ax4.annotate(name, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_title('Risk-Return Profile')
        ax4.set_xlabel('Volatility (%)')
        ax4.set_ylabel('Total Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, output_file: str = "backtest_results.json"):
        """
        Save backtest results to file.
        
        Args:
            output_file: Output file path
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for key, result in self.backtest_results.items():
            serializable_results[key] = {}
            for subkey, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key][subkey] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    serializable_results[key][subkey] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
                else:
                    serializable_results[key][subkey] = value
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Backtest results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Initialize backtester
    api_key = "your-openai-api-key-here"  # Replace with actual API key
    backtester = OpenAIBacktester(api_key)
    
    # Run OpenAI bot backtest
    print("Running OpenAI bot backtest...")
    bot_results = backtester.run_backtest(
        start_date='2015-01-01',
        end_date='2020-01-01',
        rebalance_frequency='weekly',
        verbose=True
    )
    
    # Run benchmark comparison
    print("\nRunning benchmark strategies...")
    benchmark_results = backtester.run_benchmark_comparison(
        start_date='2015-01-01',
        end_date='2020-01-01'
    )
    
    # Generate and print performance report
    report = backtester.generate_performance_report()
    print(report)
    
    # Plot performance
    backtester.plot_performance(save_path="openai_bot_performance.png")
    
    # Save results
    backtester.save_results("openai_backtest_results.json")
    
    print("\n✓ Backtesting completed successfully!")
