"""
OpenAI Portfolio Environment

A simplified portfolio environment designed to work seamlessly with OpenAI's API.
This environment converts market data into natural language descriptions that
OpenAI can understand and respond to for trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta


class OpenAIPortfolioEnv:
    """
    Portfolio environment optimized for OpenAI integration.
    
    This environment provides natural language interfaces and simplified
    state representations that work well with OpenAI's language models.
    """
    
    def __init__(self, 
                 prices_df: pd.DataFrame,
                 returns_df: pd.DataFrame,
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.3):
        """
        Initialize the OpenAI Portfolio Environment.
        
        Args:
            prices_df: DataFrame with stock prices
            returns_df: DataFrame with stock returns
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage
            max_position_size: Maximum position size per stock
        """
        self.prices_df = prices_df
        self.returns_df = returns_df
        self.stock_symbols = list(prices_df.columns)
        self.n_stocks = len(self.stock_symbols)
        
        # Environment parameters
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        
        # State variables
        self.current_step = 0
        self.portfolio_weights = np.ones(self.n_stocks) / self.n_stocks  # Equal weights initially
        self.cash_weight = 0.0
        self.portfolio_value = initial_capital
        
        # History tracking
        self.portfolio_history = []
        self.weight_history = []
        self.return_history = []
        self.trade_history = []
        
        # Performance metrics
        self.total_return = 0.0
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
    def reset(self) -> Dict:
        """Reset environment to initial state."""
        self.current_step = 0
        self.portfolio_weights = np.ones(self.n_stocks) / self.n_stocks
        self.cash_weight = 0.0
        self.portfolio_value = self.initial_capital
        
        # Clear history
        self.portfolio_history = []
        self.weight_history = []
        self.return_history = []
        self.trade_history = []
        
        return self._get_state_description()
    
    def _get_state_description(self) -> Dict:
        """
        Get current state as natural language description for OpenAI.
        
        Returns:
            Dictionary with state information in OpenAI-friendly format
        """
        if self.current_step >= len(self.returns_df):
            return {"error": "Episode completed"}
        
        # Current market data
        current_prices = self.prices_df.iloc[self.current_step].to_dict()
        current_returns = self.returns_df.iloc[self.current_step].to_dict()
        
        # Historical context (last 5 days)
        lookback = min(5, self.current_step)
        if lookback > 0:
            recent_returns = self.returns_df.iloc[self.current_step-lookback:self.current_step].mean().to_dict()
            recent_volatility = self.returns_df.iloc[self.current_step-lookback:self.current_step].std().to_dict()
        else:
            recent_returns = {symbol: 0.0 for symbol in self.stock_symbols}
            recent_volatility = {symbol: 0.0 for symbol in self.stock_symbols}
        
        # Portfolio state
        portfolio_summary = {}
        for i, symbol in enumerate(self.stock_symbols):
            portfolio_summary[symbol] = {
                "weight": self.portfolio_weights[i],
                "value": self.portfolio_weights[i] * self.portfolio_value
            }
        
        # Create natural language description
        state_description = f"""
        === PORTFOLIO STATE ===
        Current Step: {self.current_step}
        Portfolio Value: ${self.portfolio_value:.2f}
        Total Return: {((self.portfolio_value / self.initial_capital) - 1) * 100:.2f}%
        
        === MARKET DATA ===
        Current Prices:
        """
        
        for symbol in self.stock_symbols:
            price = current_prices[symbol]
            daily_return = current_returns[symbol] * 100
            recent_ret = recent_returns[symbol] * 100
            recent_vol = recent_volatility[symbol] * 100
            
            state_description += f"• {symbol}: ${price:.2f} (Today: {daily_return:+.2f}%, Avg 5-day: {recent_ret:+.2f}%, Vol: {recent_vol:.2f}%)\n"
        
        state_description += "\n=== CURRENT PORTFOLIO ===\n"
        for symbol, info in portfolio_summary.items():
            state_description += f"• {symbol}: {info['weight']:.1%} (${info['value']:.2f})\n"
        
        # Add market sentiment indicators
        state_description += "\n=== MARKET SENTIMENT ===\n"
        positive_stocks = sum(1 for ret in current_returns.values() if ret > 0)
        total_stocks = len(current_returns)
        sentiment = "Bullish" if positive_stocks > total_stocks * 0.6 else "Bearish" if positive_stocks < total_stocks * 0.4 else "Neutral"
        
        state_description += f"Market Sentiment: {sentiment} ({positive_stocks}/{total_stocks} stocks positive)\n"
        
        # Risk metrics
        if len(self.return_history) > 10:
            recent_vol = np.std(self.return_history[-10:]) * np.sqrt(252) * 100
            state_description += f"Portfolio Volatility (10-day): {recent_vol:.2f}%\n"
        
        return {
            "description": state_description,
            "current_prices": current_prices,
            "current_returns": current_returns,
            "portfolio_weights": self.portfolio_weights.tolist(),
            "portfolio_value": self.portfolio_value,
            "step": self.current_step,
            "sentiment": sentiment,
            "recent_returns": recent_returns,
            "recent_volatility": recent_volatility
        }
    
    def step(self, action_description: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment based on OpenAI's action description.
        
        Args:
            action_description: Natural language description of the action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_step >= len(self.returns_df) - 1:
            return self._get_state_description(), 0.0, True, {"message": "Episode completed"}
        
        # Parse action and update portfolio
        success = self._execute_action(action_description)
        
        # Calculate portfolio return
        if self.current_step < len(self.returns_df):
            current_returns = self.returns_df.iloc[self.current_step].values
            portfolio_return = np.sum(self.portfolio_weights * current_returns)
            self.return_history.append(portfolio_return)
            
            # Update portfolio value
            self.portfolio_value *= (1 + portfolio_return)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Store history
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(self.portfolio_weights.copy())
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.returns_df) - 1
        
        # Get next state
        next_state = self._get_state_description()
        
        # Info dictionary
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return if 'portfolio_return' in locals() else 0.0,
            "total_return": (self.portfolio_value / self.initial_capital - 1) * 100,
            "step": self.current_step,
            "action_success": success
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action_description: str) -> bool:
        """
        Execute action based on natural language description.
        
        This is a simplified action parser. In practice, you might want to
        use more sophisticated NLP techniques or ask OpenAI to provide
        structured actions.
        
        Args:
            action_description: Natural language description of the action
            
        Returns:
            Boolean indicating if action was executed successfully
        """
        action_lower = action_description.lower()
        
        # Simple keyword-based action parsing
        if "buy" in action_lower or "increase" in action_lower:
            return self._execute_buy_action(action_description)
        elif "sell" in action_lower or "decrease" in action_lower:
            return self._execute_sell_action(action_description)
        elif "hold" in action_lower or "maintain" in action_lower:
            return self._execute_hold_action(action_description)
        else:
            # Default to hold if action is unclear
            return self._execute_hold_action("hold current positions")
    
    def _execute_buy_action(self, action_description: str) -> bool:
        """Execute buy action based on description."""
        # Extract stock symbols mentioned in the action
        mentioned_stocks = []
        for symbol in self.stock_symbols:
            if symbol.lower() in action_description.lower():
                mentioned_stocks.append(symbol)
        
        if not mentioned_stocks:
            return False
        
        # Simple strategy: increase weight of mentioned stocks
        adjustment = 0.05  # 5% adjustment per action
        
        for symbol in mentioned_stocks:
            stock_idx = self.stock_symbols.index(symbol)
            current_weight = self.portfolio_weights[stock_idx]
            
            # Increase weight but respect maximum position size
            new_weight = min(current_weight + adjustment, self.max_position_size)
            
            # Apply transaction cost
            weight_change = new_weight - current_weight
            transaction_penalty = weight_change * self.transaction_cost
            new_weight -= transaction_penalty
            
            self.portfolio_weights[stock_idx] = new_weight
        
        # Renormalize weights to sum to 1
        self._normalize_weights()
        
        # Record trade
        self.trade_history.append({
            "step": self.current_step,
            "action": "buy",
            "description": action_description,
            "stocks": mentioned_stocks,
            "new_weights": self.portfolio_weights.copy()
        })
        
        return True
    
    def _execute_sell_action(self, action_description: str) -> bool:
        """Execute sell action based on description."""
        # Extract stock symbols mentioned in the action
        mentioned_stocks = []
        for symbol in self.stock_symbols:
            if symbol.lower() in action_description.lower():
                mentioned_stocks.append(symbol)
        
        if not mentioned_stocks:
            return False
        
        # Simple strategy: decrease weight of mentioned stocks
        adjustment = 0.05  # 5% adjustment per action
        
        for symbol in mentioned_stocks:
            stock_idx = self.stock_symbols.index(symbol)
            current_weight = self.portfolio_weights[stock_idx]
            
            # Decrease weight but don't go below 0
            new_weight = max(current_weight - adjustment, 0.0)
            
            # Apply transaction cost
            weight_change = current_weight - new_weight
            transaction_penalty = weight_change * self.transaction_cost
            new_weight -= transaction_penalty
            
            self.portfolio_weights[stock_idx] = new_weight
        
        # Renormalize weights to sum to 1
        self._normalize_weights()
        
        # Record trade
        self.trade_history.append({
            "step": self.current_step,
            "action": "sell",
            "description": action_description,
            "stocks": mentioned_stocks,
            "new_weights": self.portfolio_weights.copy()
        })
        
        return True
    
    def _execute_hold_action(self, action_description: str) -> bool:
        """Execute hold action (no changes to portfolio)."""
        self.trade_history.append({
            "step": self.current_step,
            "action": "hold",
            "description": action_description,
            "stocks": [],
            "new_weights": self.portfolio_weights.copy()
        })
        return True
    
    def _normalize_weights(self):
        """Normalize portfolio weights to sum to 1."""
        total_weight = np.sum(self.portfolio_weights)
        if total_weight > 0:
            self.portfolio_weights = self.portfolio_weights / total_weight
        else:
            # If all weights are zero, reset to equal weights
            self.portfolio_weights = np.ones(self.n_stocks) / self.n_stocks
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on portfolio performance.
        
        Returns:
            Reward value
        """
        if len(self.return_history) < 2:
            return 0.0
        
        # Portfolio return
        portfolio_return = self.return_history[-1]
        
        # Risk-adjusted reward (Sharpe-like)
        if len(self.return_history) > 10:
            recent_returns = np.array(self.return_history[-10:])
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return > 0:
                sharpe_ratio = mean_return / std_return
                reward = sharpe_ratio * 0.1  # Scale down
            else:
                reward = portfolio_return
        else:
            reward = portfolio_return
        
        # Transaction cost penalty
        if len(self.trade_history) > 0:
            last_trade = self.trade_history[-1]
            if last_trade["action"] != "hold":
                reward -= 0.001  # Small penalty for trading
        
        return reward
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(self.portfolio_history) < 2:
            return {}
        
        # Total return
        total_return = (self.portfolio_value / self.initial_capital - 1) * 100
        
        # Volatility
        if len(self.return_history) > 1:
            volatility = np.std(self.return_history) * np.sqrt(252) * 100
        else:
            volatility = 0.0
        
        # Sharpe ratio
        if volatility > 0:
            mean_return = np.mean(self.return_history) * 252  # Annualized
            sharpe_ratio = mean_return / (volatility / 100)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(self.portfolio_history)
        drawdown = (np.array(self.portfolio_history) - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Trade statistics
        total_trades = len(self.trade_history)
        buy_trades = sum(1 for trade in self.trade_history if trade["action"] == "buy")
        sell_trades = sum(1 for trade in self.trade_history if trade["action"] == "sell")
        hold_trades = sum(1 for trade in self.trade_history if trade["action"] == "hold")
        
        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "hold_trades": hold_trades,
            "final_portfolio_value": self.portfolio_value,
            "final_weights": dict(zip(self.stock_symbols, self.portfolio_weights))
        }
    
    def render(self, mode: str = "human"):
        """Render the current state of the environment."""
        if mode == "human":
            state = self._get_state_description()
            print(state["description"])
            
            metrics = self.get_performance_metrics()
            if metrics:
                print("\n=== PERFORMANCE METRICS ===")
                print(f"Total Return: {metrics['total_return']:.2f}%")
                print(f"Volatility: {metrics['volatility']:.2f}%")
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
                print(f"Total Trades: {metrics['total_trades']}")
    
    def get_portfolio_summary(self) -> str:
        """Get portfolio summary as natural language for OpenAI."""
        metrics = self.get_performance_metrics()
        
        summary = f"""
        Portfolio Summary:
        - Current Value: ${self.portfolio_value:.2f}
        - Total Return: {metrics.get('total_return', 0):.2f}%
        - Volatility: {metrics.get('volatility', 0):.2f}%
        - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        - Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
        - Total Trades: {metrics.get('total_trades', 0)}
        
        Current Allocation:
        """
        
        for symbol, weight in zip(self.stock_symbols, self.portfolio_weights):
            summary += f"- {symbol}: {weight:.1%}\n"
        
        return summary


def create_openai_env_from_data(prices_df: pd.DataFrame, 
                               returns_df: pd.DataFrame,
                               **kwargs) -> OpenAIPortfolioEnv:
    """
    Create OpenAI Portfolio Environment from market data.
    
    Args:
        prices_df: DataFrame with stock prices
        returns_df: DataFrame with stock returns
        **kwargs: Additional environment parameters
        
    Returns:
        Initialized OpenAIPortfolioEnv instance
    """
    return OpenAIPortfolioEnv(prices_df, returns_df, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    stocks = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'JNJ']
    
    np.random.seed(42)
    base_prices = {'AAPL': 150, 'AMZN': 3000, 'GOOG': 2500, 'MSFT': 300, 'JNJ': 160}
    
    price_data = {}
    for stock in stocks:
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_prices[stock]]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        price_data[stock] = prices[1:]
    
    prices_df = pd.DataFrame(price_data, index=dates)
    returns_df = prices_df.pct_change().dropna()
    
    # Create environment
    env = OpenAIPortfolioEnv(prices_df, returns_df)
    
    # Reset and run a few steps
    state = env.reset()
    print("Initial State:")
    print(state["description"])
    
    # Run a few steps with sample actions
    actions = [
        "Buy more AAPL and AMZN stocks",
        "Hold current positions",
        "Sell some GOOG and MSFT",
        "Hold positions"
    ]
    
    for i, action in enumerate(actions):
        print(f"\n--- Step {i+1}: {action} ---")
        next_state, reward, done, info = env.step(action)
        
        print(f"Reward: {reward:.4f}")
        print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
        
        if done:
            break
    
    # Show final performance
    print("\n=== FINAL PERFORMANCE ===")
    env.render()
