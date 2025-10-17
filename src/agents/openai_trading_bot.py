"""
OpenAI-Powered Trading Bot

This module implements a trading bot using OpenAI's GPT-4 API for portfolio optimization
without traditional reinforcement learning. Instead, it uses OpenAI's function calling
capabilities and natural language processing for trading decisions.
"""

import openai
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime, timedelta
import time


class OpenAITradingBot:
    """
    Trading bot powered by OpenAI's GPT-4 for portfolio optimization.
    
    This bot uses OpenAI's function calling capabilities to make trading decisions
    based on market data and portfolio state.
    """
    
    def __init__(self, api_key: str, initial_capital: float = 10000.0):
        """
        Initialize the OpenAI Trading Bot.
        
        Args:
            api_key: OpenAI API key
            initial_capital: Starting capital for the portfolio
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio = {}  # {symbol: shares}
        self.cash = initial_capital
        self.trade_history = []
        self.portfolio_values = [initial_capital]
        
        # Trading parameters
        self.max_position_size = 0.3  # Maximum 30% in any single stock
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.min_trade_amount = 100  # Minimum trade size
        
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary for OpenAI analysis."""
        portfolio_value = self.cash
        holdings = {}
        
        for symbol, shares in self.portfolio.items():
            # This would normally get current price from market data
            # For now, we'll use a placeholder
            current_price = 100.0  # Placeholder
            position_value = shares * current_price
            portfolio_value += position_value
            holdings[symbol] = {
                'shares': shares,
                'current_price': current_price,
                'position_value': position_value,
                'percentage': position_value / portfolio_value
            }
        
        return {
            'total_value': portfolio_value,
            'cash': self.cash,
            'cash_percentage': self.cash / portfolio_value,
            'holdings': holdings,
            'total_return': (portfolio_value / self.initial_capital - 1) * 100
        }
    
    def format_market_data(self, market_data: pd.DataFrame) -> str:
        """Format market data for OpenAI consumption."""
        market_summary = "Current Market Data:\n"
        
        for symbol in market_data.columns:
            if symbol in market_data.columns:
                current_price = market_data[symbol].iloc[-1]
                daily_change = market_data[symbol].pct_change().iloc[-1] * 100
                
                market_summary += f"- {symbol}: ${current_price:.2f} "
                market_summary += f"({daily_change:+.2f}%)\n"
        
        return market_summary
    
    def get_trading_recommendation(self, market_data: pd.DataFrame) -> Dict:
        """
        Get trading recommendation from OpenAI GPT-4.
        
        Args:
            market_data: DataFrame with stock prices
            
        Returns:
            Dictionary with trading recommendation
        """
        portfolio_summary = self.get_portfolio_summary()
        market_summary = self.format_market_data(market_data)
        
        system_prompt = """You are an expert portfolio manager with deep knowledge of financial markets. 
        Your task is to analyze market data and current portfolio state to make optimal trading decisions.
        
        Consider:
        1. Risk management (don't put more than 30% in any single stock)
        2. Diversification across sectors
        3. Market trends and momentum
        4. Current portfolio allocation
        5. Transaction costs (0.1% per trade)
        
        Provide clear reasoning for your decisions."""
        
        user_prompt = f"""
        {market_summary}
        
        Current Portfolio:
        Total Value: ${portfolio_summary['total_value']:.2f}
        Cash: ${portfolio_summary['cash']:.2f} ({portfolio_summary['cash_percentage']:.1f}%)
        Total Return: {portfolio_summary['total_return']:.2f}%
        
        Current Holdings:
        {json.dumps(portfolio_summary['holdings'], indent=2)}
        
        Please analyze this situation and provide trading recommendations.
        """
        
        # Define functions that GPT can call
        functions = [
            {
                "name": "buy_stock",
                "description": "Buy shares of a stock",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol to buy"
                        },
                        "amount": {
                            "type": "number",
                            "description": "Dollar amount to invest"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reason for buying this stock"
                        }
                    },
                    "required": ["symbol", "amount", "reasoning"]
                }
            },
            {
                "name": "sell_stock",
                "description": "Sell shares of a stock",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol to sell"
                        },
                        "percentage": {
                            "type": "number",
                            "description": "Percentage of position to sell (0-100)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reason for selling this stock"
                        }
                    },
                    "required": ["symbol", "percentage", "reasoning"]
                }
            },
            {
                "name": "hold_position",
                "description": "Hold current position without changes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Reason for holding current positions"
                        }
                    },
                    "required": ["reasoning"]
                }
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                functions=functions,
                function_call="auto",
                temperature=0.3  # Lower temperature for more consistent decisions
            )
            
            return self._parse_openai_response(response)
            
        except Exception as e:
            print(f"Error getting OpenAI recommendation: {e}")
            return {"action": "hold", "reasoning": "Error in analysis", "details": {}}
    
    def _parse_openai_response(self, response) -> Dict:
        """Parse OpenAI response and extract trading recommendation."""
        message = response.choices[0].message
        
        if message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)
            
            return {
                "action": function_name,
                "reasoning": function_args.get("reasoning", ""),
                "details": function_args
            }
        else:
            return {
                "action": "hold",
                "reasoning": message.content,
                "details": {}
            }
    
    def execute_trade(self, recommendation: Dict, current_prices: Dict[str, float]) -> bool:
        """
        Execute the trading recommendation.
        
        Args:
            recommendation: Trading recommendation from OpenAI
            current_prices: Dictionary with current stock prices
            
        Returns:
            Boolean indicating if trade was executed successfully
        """
        action = recommendation["action"]
        details = recommendation["details"]
        
        try:
            if action == "buy_stock":
                symbol = details["symbol"]
                amount = details["amount"]
                
                # Validate trade
                if amount < self.min_trade_amount:
                    print(f"Trade amount ${amount} below minimum ${self.min_trade_amount}")
                    return False
                
                if amount > self.cash * 0.95:  # Leave 5% cash buffer
                    print(f"Insufficient cash for trade: ${amount}")
                    return False
                
                # Calculate shares to buy
                price = current_prices.get(symbol, 0)
                if price == 0:
                    print(f"No price data for {symbol}")
                    return False
                
                shares_to_buy = (amount * (1 - self.transaction_cost)) / price
                
                # Execute trade
                self.cash -= amount
                if symbol in self.portfolio:
                    self.portfolio[symbol] += shares_to_buy
                else:
                    self.portfolio[symbol] = shares_to_buy
                
                # Record trade
                self.trade_history.append({
                    "timestamp": datetime.now(),
                    "action": "buy",
                    "symbol": symbol,
                    "amount": amount,
                    "shares": shares_to_buy,
                    "price": price,
                    "reasoning": recommendation["reasoning"]
                })
                
                print(f"Bought {shares_to_buy:.2f} shares of {symbol} for ${amount:.2f}")
                return True
                
            elif action == "sell_stock":
                symbol = details["symbol"]
                percentage = details["percentage"]
                
                if symbol not in self.portfolio:
                    print(f"No position in {symbol} to sell")
                    return False
                
                # Calculate shares to sell
                shares_to_sell = self.portfolio[symbol] * (percentage / 100)
                price = current_prices.get(symbol, 0)
                
                if price == 0:
                    print(f"No price data for {symbol}")
                    return False
                
                sale_amount = shares_to_sell * price * (1 - self.transaction_cost)
                
                # Execute trade
                self.portfolio[symbol] -= shares_to_sell
                if self.portfolio[symbol] <= 0:
                    del self.portfolio[symbol]
                
                self.cash += sale_amount
                
                # Record trade
                self.trade_history.append({
                    "timestamp": datetime.now(),
                    "action": "sell",
                    "symbol": symbol,
                    "amount": sale_amount,
                    "shares": shares_to_sell,
                    "price": price,
                    "reasoning": recommendation["reasoning"]
                })
                
                print(f"Sold {shares_to_sell:.2f} shares of {symbol} for ${sale_amount:.2f}")
                return True
                
            elif action == "hold_position":
                print(f"Holding positions: {recommendation['reasoning']}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value based on current prices."""
        portfolio_value = self.cash
        
        for symbol, shares in self.portfolio.items():
            price = current_prices.get(symbol, 0)
            portfolio_value += shares * price
        
        self.portfolio_values.append(portfolio_value)
        return portfolio_value
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.portfolio_values) < 2:
            return {}
        
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        total_return = (self.portfolio_values[-1] / self.initial_capital - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        peak = pd.Series(self.portfolio_values).expanding().max()
        drawdown = (pd.Series(self.portfolio_values) - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trade_history),
            "current_value": self.portfolio_values[-1]
        }
    
    def run_trading_session(self, market_data: pd.DataFrame, current_prices: Dict[str, float]):
        """
        Run a single trading session.
        
        Args:
            market_data: Historical market data
            current_prices: Current stock prices
        """
        # Get trading recommendation
        recommendation = self.get_trading_recommendation(market_data)
        
        # Execute trade
        success = self.execute_trade(recommendation, current_prices)
        
        # Update portfolio value
        portfolio_value = self.update_portfolio_value(current_prices)
        
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Holdings: {self.portfolio}")
        print("-" * 50)
        
        return success


def create_sample_market_data() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Create sample market data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    stocks = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'JNJ']
    
    # Generate sample price data
    np.random.seed(42)
    base_prices = {'AAPL': 150, 'AMZN': 3000, 'GOOG': 2500, 'MSFT': 300, 'JNJ': 160}
    
    price_data = {}
    for stock in stocks:
        returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
        prices = [base_prices[stock]]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        price_data[stock] = prices[1:]  # Remove initial price
    
    market_df = pd.DataFrame(price_data, index=dates)
    
    # Current prices (last day)
    current_prices = {stock: market_df[stock].iloc[-1] for stock in stocks}
    
    return market_df, current_prices


if __name__ == "__main__":
    # Example usage
    print("OpenAI Trading Bot Demo")
    print("=" * 50)
    
    # Initialize bot (you'll need to set your API key)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    bot = OpenAITradingBot(api_key)
    
    # Create sample data
    market_data, current_prices = create_sample_market_data()
    
    # Run trading session
    bot.run_trading_session(market_data, current_prices)
    
    # Show performance
    metrics = bot.get_performance_metrics()
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
