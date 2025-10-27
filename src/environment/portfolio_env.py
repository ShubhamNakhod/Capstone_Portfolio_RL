"""
Enhanced Portfolio Environment for Reinforcement Learning

This module implements a comprehensive portfolio optimization environment that incorporates
all insights from the EDA analysis, including technical indicators, correlation features,
market regime indicators, and risk metrics.

Based on the analysis in notebooks/02_Environment_Design.ipynb
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class BasePortfolioEnv(gym.Env):
    """
    Base Portfolio Environment for Reinforcement Learning
    
    This is a base environment that can be extended with different
    state representations, action spaces, and reward functions.
    """
    
    def __init__(self, 
                 prices_df: pd.DataFrame, 
                 returns_df: pd.DataFrame, 
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001, 
                 episode_length: int = 252, 
                 lookback_window: int = 20):
        super().__init__()
        
        # Data
        self.prices_df = prices_df
        self.returns_df = returns_df
        self.stock_names = list(prices_df.columns)
        self.n_stocks = len(self.stock_names)
        
        # Environment parameters
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.episode_length = episode_length
        self.lookback_window = lookback_window
        
        # Episode state
        self.current_step = 0
        self.current_portfolio_value = initial_capital
        self.portfolio_weights = np.ones(self.n_stocks) / self.n_stocks  # Equal weights initially
        self.cash_weight = 0.0
        
        # Track performance
        self.portfolio_history = []
        self.weight_history = []
        self.return_history = []
        
        # Define action and observation spaces (to be overridden in subclasses)
        self.action_space = None  # Will be defined in subclasses
        self.observation_space = None  # Will be defined in subclasses
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = self.lookback_window  # Start after lookback window
        self.current_portfolio_value = self.initial_capital
        self.portfolio_weights = np.ones(self.n_stocks) / self.n_stocks
        self.cash_weight = 0.0
        
        # Clear history
        self.portfolio_history = []
        self.weight_history = []
        self.return_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Execute action (to be implemented in subclasses)
        self._execute_action(action)
        
        # Calculate reward (to be implemented in subclasses)
        reward = self._calculate_reward()
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Store history
        self.portfolio_history.append(self.current_portfolio_value)
        self.weight_history.append(self.portfolio_weights.copy())
        
        # Check if episode is done
        self.current_step += 1
        done = (self.current_step >= len(self.returns_df) - 1) or (self.current_step >= self.episode_length)
        
        # Get next observation
        observation = self._get_observation() if not done else None
        info = self._get_info()
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        """Get current observation (to be implemented in subclasses)"""
        raise NotImplementedError("Subclasses must implement _get_observation")
    
    def _execute_action(self, action):
        """Execute action (to be implemented in subclasses)"""
        raise NotImplementedError("Subclasses must implement _execute_action")
    
    def _calculate_reward(self):
        """Calculate reward (to be implemented in subclasses)"""
        raise NotImplementedError("Subclasses must implement _calculate_reward")
    
    def _update_portfolio_value(self):
        """Update portfolio value based on returns"""
        if self.current_step < len(self.returns_df):
            # Calculate portfolio return
            current_returns = self.returns_df.iloc[self.current_step].values
            portfolio_return = np.sum(self.portfolio_weights * current_returns)
            
            # Update portfolio value
            self.current_portfolio_value *= (1 + portfolio_return)
            self.return_history.append(portfolio_return)
    
    def _get_info(self):
        """Get additional information"""
        return {
            'portfolio_value': self.current_portfolio_value,
            'portfolio_weights': self.portfolio_weights.copy(),
            'current_step': self.current_step,
            'total_return': (self.current_portfolio_value / self.initial_capital) - 1
        }
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.current_portfolio_value:.2f}")
            print(f"Total Return: {((self.current_portfolio_value / self.initial_capital) - 1) * 100:.2f}%")
            print(f"Portfolio Weights: {dict(zip(self.stock_names, self.portfolio_weights))}")


class EnhancedPortfolioEnv(BasePortfolioEnv):
    """
    Enhanced Portfolio Environment incorporating EDA insights:
    - Complete technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands)
    - Correlation features and diversification metrics
    - Risk metrics (VaR, max drawdown, risk-free rate)
    - Market regime indicators
    - Enhanced reward function
    """
    
    def __init__(self, prices_df, returns_df, **kwargs):
        super().__init__(prices_df, returns_df, **kwargs)
        
        # Calculate enhanced technical indicators
        self._calculate_enhanced_indicators()
        
        # Calculate correlation features
        self._calculate_correlation_features()
        
        # Calculate market regime indicators
        self._calculate_market_regime()
        
        # Enhanced state space: weights + technical + correlation + regime + risk
        n_corr_pairs = self.n_stocks * (self.n_stocks - 1) // 2
        
        state_dim = (self.n_stocks +  # portfolio weights
                    (self.n_stocks * 6) +  # technical indicators per stock
                    n_corr_pairs +  # correlation features (upper triangular)
                    3 +  # market regime indicators
                    self.n_stocks)  # risk metrics per stock
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Continuous action space with softmax
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(self.n_stocks,), dtype=np.float32
        )
        
        # Risk-free rate from EDA analysis
        self.risk_free_rate = 0.02
        
    def _calculate_enhanced_indicators(self):
        """Calculate comprehensive technical indicators"""
        self.indicators = {}
        
        for stock in self.stock_names:
            prices = self.prices_df[stock].values
            returns = self.returns_df[stock].values
            
            # Price-based indicators
            price_series = pd.Series(prices)
            return_series = pd.Series(returns)
            
            # 1. RSI (proper calculation)
            rsi = self._calculate_rsi(return_series, period=14)
            
            # 2. MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(price_series)
            
            # 3. Moving Averages
            sma_20 = price_series.rolling(20).mean()
            ema_12 = price_series.ewm(span=12).mean()
            
            # 4. Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_series)
            
            # 5. Price ratios
            price_ratio = prices / sma_20.values
            
            # 6. Volatility (20-day rolling)
            volatility = return_series.rolling(20).std()
            
            self.indicators[stock] = {
                'rsi': rsi.values,
                'macd': macd_line.values,
                'macd_signal': macd_signal.values,
                'macd_histogram': macd_histogram.values,
                'sma_20': sma_20.values,
                'ema_12': ema_12.values,
                'bb_upper': bb_upper.values,
                'bb_middle': bb_middle.values,
                'bb_lower': bb_lower.values,
                'price_ratio': price_ratio,  # Already numpy array
                'volatility': volatility.values
            }
    
    def _calculate_rsi(self, returns, period=14):
        """Calculate proper RSI"""
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Neutral RSI for NaN values
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicators"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def _calculate_correlation_features(self):
        """Calculate correlation-based features for diversification"""
        # Calculate rolling correlation matrix (20-day window)
        # We'll store the upper triangular part of the correlation matrix
        self.correlation_features = {}
        
        # Create a list to store correlation pairs
        correlation_pairs = []
        
        for i, stock1 in enumerate(self.stock_names):
            for j, stock2 in enumerate(self.stock_names):
                if i < j:  # Only upper triangular part to avoid duplicates
                    # Rolling correlation
                    corr = self.returns_df[stock1].rolling(20).corr(self.returns_df[stock2])
                    # Fill NaN values with 0 (no correlation)
                    corr_filled = corr.fillna(0).values
                    correlation_pairs.append(corr_filled)
        
        # Store as a single array
        self.correlation_features = np.array(correlation_pairs).T
    
    def _calculate_market_regime(self):
        """Calculate market regime indicators"""
        # 1. Volatility regime (high/low volatility)
        market_volatility = self.returns_df.mean(axis=1).rolling(20).std()
        vol_regime = (market_volatility > market_volatility.rolling(60).mean()).astype(int)
        
        # 2. Trend regime (bull/bear market)
        market_returns = self.returns_df.mean(axis=1)
        trend_regime = (market_returns.rolling(60).sum() > 0).astype(int)
        
        # 3. Cycle position (early/late cycle based on volatility patterns)
        vol_cycle = market_volatility.rolling(60).rank(pct=True)
        
        # Fill NaN values with defaults
        vol_regime = vol_regime.fillna(0).values
        trend_regime = trend_regime.fillna(0).values
        vol_cycle = vol_cycle.fillna(0.5).values  # Neutral cycle position
        
        self.market_regime = {
            'volatility_regime': vol_regime,
            'trend_regime': trend_regime,
            'cycle_position': vol_cycle
        }
    
    def _get_observation(self):
        """Get enhanced observation"""
        obs = []
        
        # 1. Current portfolio weights
        obs.extend(self.portfolio_weights)
        
        # 2. Technical indicators for current step
        for stock in self.stock_names:
            if self.current_step < len(self.prices_df):
                indicators = self.indicators[stock]
                # Handle NaN values with defaults
                rsi_val = indicators['rsi'][self.current_step]
                macd_val = indicators['macd'][self.current_step]
                macd_sig_val = indicators['macd_signal'][self.current_step]
                price_ratio_val = indicators['price_ratio'][self.current_step]
                vol_val = indicators['volatility'][self.current_step]
                
                obs.append(50.0 if np.isnan(rsi_val) else rsi_val)
                obs.append(0.0 if np.isnan(macd_val) else macd_val)
                obs.append(0.0 if np.isnan(macd_sig_val) else macd_sig_val)
                obs.append(1.0 if np.isnan(price_ratio_val) else price_ratio_val)
                obs.append(0.0 if np.isnan(vol_val) else vol_val)
                
                # BB squeeze (avoid division by zero)
                bb_upper = indicators['bb_upper'][self.current_step]
                bb_lower = indicators['bb_lower'][self.current_step]
                if np.isnan(bb_upper) or np.isnan(bb_lower) or bb_lower <= 0:
                    bb_squeeze = 1.0
                else:
                    bb_squeeze = bb_upper / bb_lower
                obs.append(bb_squeeze)
            else:
                obs.extend([50.0, 0.0, 0.0, 1.0, 0.0, 1.0])  # Default values
        
        # 3. Correlation features (upper triangular part of correlation matrix)
        if self.current_step < len(self.returns_df):
            corr_features = self.correlation_features[self.current_step]
            # Handle NaN values in correlations
            corr_features_clean = [0.0 if np.isnan(x) else x for x in corr_features]
            obs.extend(corr_features_clean)
        else:
            # Number of correlation pairs = n_stocks * (n_stocks - 1) / 2
            n_corr_pairs = self.n_stocks * (self.n_stocks - 1) // 2
            obs.extend([0.0] * n_corr_pairs)
        
        # 4. Market regime indicators
        if self.current_step < len(self.returns_df):
            vol_regime = self.market_regime['volatility_regime'][self.current_step]
            trend_regime = self.market_regime['trend_regime'][self.current_step]
            cycle_pos = self.market_regime['cycle_position'][self.current_step]
            
            # Handle NaN values
            obs.append(0.0 if np.isnan(vol_regime) else vol_regime)
            obs.append(0.0 if np.isnan(trend_regime) else trend_regime)
            obs.append(0.5 if np.isnan(cycle_pos) else cycle_pos)
        else:
            obs.extend([0.0, 0.0, 0.5])
        
        # 5. Risk metrics per stock
        for stock in self.stock_names:
            if self.current_step >= 20:  # Need history for VaR
                recent_returns = self.returns_df[stock].iloc[max(0, self.current_step-20):self.current_step]
                # Remove NaN values before calculating VaR
                clean_returns = recent_returns.dropna()
                if len(clean_returns) > 0:
                    var_95 = np.percentile(clean_returns, 5)
                    obs.append(-0.05 if np.isnan(var_95) else var_95)
                else:
                    obs.append(-0.05)  # Default VaR if no clean data
            else:
                obs.append(-0.05)  # Default VaR
        
        return np.array(obs, dtype=np.float32)
    
    def _execute_action(self, action):
        """Execute action with enhanced constraints"""
        # Convert logits to portfolio weights using softmax
        action_logits = np.array(action)
        exp_logits = np.exp(action_logits - np.max(action_logits))
        new_weights = exp_logits / np.sum(exp_logits)
        
        # Apply diversification constraints based on correlation analysis
        # Penalize high correlation with other stocks
        correlation_penalty = 0
        if self.current_step < len(self.returns_df):
            # Get correlation features for current step
            corr_features = self.correlation_features[self.current_step]
            corr_idx = 0
            
            # Ensure we don't exceed bounds
            n_stocks_in_action = len(new_weights)
            n_stocks_in_env = len(self.stock_names)
            
            for i, stock1 in enumerate(self.stock_names):
                for j, stock2 in enumerate(self.stock_names):
                    if i < j:  # Only upper triangular part
                        # Check bounds to prevent index errors
                        if i < n_stocks_in_action and j < n_stocks_in_action:
                            corr = corr_features[corr_idx]
                            if abs(corr) > 0.7:  # High correlation threshold from EDA
                                correlation_penalty += new_weights[i] * new_weights[j] * abs(corr) * 0.1
                        corr_idx += 1
        
        # Store penalties for reward calculation
        self.last_correlation_penalty = correlation_penalty
        
        # Handle dimension mismatch between new_weights and portfolio_weights
        # This can happen when action_dim doesn't match n_stocks (e.g., DQN with discrete strategies)
        if len(new_weights) != len(self.portfolio_weights):
            # If dimensions don't match, pad or truncate new_weights to match
            if len(new_weights) < len(self.portfolio_weights):
                # Pad with zeros
                padded_weights = np.pad(new_weights, (0, len(self.portfolio_weights) - len(new_weights)))
            else:
                # Truncate
                padded_weights = new_weights[:len(self.portfolio_weights)]
            
            # Renormalize
            padded_weights = padded_weights / np.sum(padded_weights) if np.sum(padded_weights) > 0 else self.portfolio_weights
            new_weights = padded_weights
        
        # Calculate transaction costs
        weight_change = np.sum(np.abs(new_weights - self.portfolio_weights))
        self.last_transaction_cost = weight_change * self.transaction_cost
        
        # Update weights
        self.portfolio_weights = new_weights
    
    def _calculate_reward(self):
        """Enhanced reward function incorporating EDA insights"""
        if self.current_step == 0:
            return 0
        
        # Portfolio return
        portfolio_return = self.return_history[-1] if self.return_history else 0
        
        # Risk-adjusted reward with proper risk-free rate
        if len(self.return_history) > 10:
            recent_returns = np.array(self.return_history[-10:])
            excess_return = np.mean(recent_returns) - (self.risk_free_rate / 252)  # Daily risk-free rate
            portfolio_volatility = np.std(recent_returns)
            
            if portfolio_volatility > 0:
                sharpe_ratio = excess_return / portfolio_volatility
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = portfolio_return * 10  # Scale up for early episodes
        
        # VaR penalty (penalize if portfolio VaR is too high)
        var_penalty = 0
        if self.current_step >= 20:
            portfolio_var = np.percentile(self.return_history[-20:], 5) if len(self.return_history) >= 20 else 0
            if portfolio_var < -0.05:  # VaR worse than -5%
                var_penalty = abs(portfolio_var + 0.05) * 2
        
        # Drawdown penalty
        drawdown_penalty = 0
        if len(self.portfolio_history) > 1:
            current_value = self.current_portfolio_value
            peak_value = max(self.portfolio_history)
            drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
            if drawdown > 0.1:  # 10% drawdown threshold
                drawdown_penalty = drawdown * 0.5
        
        # Market regime bonus/penalty
        regime_bonus = 0
        if self.current_step < len(self.returns_df):
            vol_regime = self.market_regime['volatility_regime'][self.current_step]
            trend_regime = self.market_regime['trend_regime'][self.current_step]
            
            # Bonus for good performance in different regimes
            if vol_regime == 1 and portfolio_return > 0:  # High vol, positive return
                regime_bonus += 0.01
            if trend_regime == 1 and portfolio_return > 0:  # Bull market, positive return
                regime_bonus += 0.005
        
        # Combine all components
        reward = (sharpe_ratio * 0.1 + 
                 regime_bonus - 
                 getattr(self, 'last_transaction_cost', 0) - 
                 getattr(self, 'last_correlation_penalty', 0) - 
                 var_penalty - 
                 drawdown_penalty)
        
        return reward


class ExpandedEnhancedPortfolioEnv(EnhancedPortfolioEnv):
    """
    Expanded Enhanced Portfolio Environment with diversified stock selection.
    
    This extends the EnhancedPortfolioEnv to use more stocks across different sectors
    for better diversification while maintaining all the EDA insights.
    """
    
    def __init__(self, prices_df, returns_df, **kwargs):
        # Initialize with expanded stock universe
        super().__init__(prices_df, returns_df, **kwargs)
        
        # Override stock names to match expanded selection
        self.stock_names = list(prices_df.columns)
        self.n_stocks = len(self.stock_names)
        
        # Recalculate state dimension for expanded universe
        n_corr_pairs = self.n_stocks * (self.n_stocks - 1) // 2
        
        state_dim = (self.n_stocks +  # portfolio weights
                    (self.n_stocks * 6) +  # technical indicators per stock
                    n_corr_pairs +  # correlation features (upper triangular)
                    3 +  # market regime indicators
                    self.n_stocks)  # risk metrics per stock
        
        # Update observation space for expanded universe
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Update action space for expanded universe
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(self.n_stocks,), dtype=np.float32
        )
        
        print(f"Expanded environment initialized with {self.n_stocks} stocks")
        print(f"State dimension: {state_dim}")
    
    def get_sector_weights(self):
        """Get current portfolio weights grouped by sector"""
        sector_weights = {
            'Technology': 0.0,
            'Financial': 0.0,
            'Healthcare': 0.0,
            'Airlines': 0.0,
            'Index': 0.0,
            'Other': 0.0
        }
        
        # Define sector mapping
        sector_mapping = {
            'AAPL': 'Technology', 'AMZN': 'Technology', 'GOOG': 'Technology', 
            'MSFT': 'Technology', 'FB': 'Technology', 'IBM': 'Technology',
            'GS': 'Financial', 'MS': 'Financial', 'WFC': 'Financial', 'BCS': 'Financial',
            'JNJ': 'Healthcare', 'MRK': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'AAL': 'Airlines', 'DAL': 'Airlines', 'LUV': 'Airlines',
            'SP500': 'Index'
        }
        
        for i, stock in enumerate(self.stock_names):
            sector = sector_mapping.get(stock, 'Other')
            sector_weights[sector] += self.portfolio_weights[i]
        
        return sector_weights
    
    def get_diversification_metrics(self):
        """Calculate diversification metrics for the expanded portfolio"""
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        hhi = np.sum(self.portfolio_weights ** 2)
        
        # Calculate effective number of stocks (1/HHI)
        effective_n_stocks = 1 / hhi if hhi > 0 else 0
        
        # Get sector weights
        sector_weights = self.get_sector_weights()
        
        # Calculate sector concentration
        sector_hhi = sum(w**2 for w in sector_weights.values())
        effective_n_sectors = 1 / sector_hhi if sector_hhi > 0 else 0
        
        return {
            'hhi': hhi,
            'effective_n_stocks': effective_n_stocks,
            'sector_weights': sector_weights,
            'sector_hhi': sector_hhi,
            'effective_n_sectors': effective_n_sectors
        }
    
    def render(self, mode='human'):
        """Enhanced render method with diversification metrics"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.current_portfolio_value:.2f}")
            print(f"Total Return: {((self.current_portfolio_value / self.initial_capital) - 1) * 100:.2f}%")
            
            # Show diversification metrics
            div_metrics = self.get_diversification_metrics()
            print(f"Effective Number of Stocks: {div_metrics['effective_n_stocks']:.2f}")
            print(f"Effective Number of Sectors: {div_metrics['effective_n_sectors']:.2f}")
            
            # Show sector allocation
            print("Sector Allocation:")
            for sector, weight in div_metrics['sector_weights'].items():
                if weight > 0.01:  # Only show sectors with >1% allocation
                    print(f"  {sector}: {weight*100:.1f}%")
            
            # Show top holdings
            print("Top Holdings:")
            stock_weights = list(zip(self.stock_names, self.portfolio_weights))
            stock_weights.sort(key=lambda x: x[1], reverse=True)
            for stock, weight in stock_weights[:5]:  # Top 5 holdings
                if weight > 0.01:  # Only show holdings >1%
                    print(f"  {stock}: {weight*100:.1f}%")


# Factory function for easy environment creation
def create_portfolio_env(env_type: str = "enhanced", 
                        prices_df: pd.DataFrame = None, 
                        returns_df: pd.DataFrame = None,
                        **kwargs) -> BasePortfolioEnv:
    """
    Factory function to create portfolio environments.
    
    Args:
        env_type: Type of environment ("enhanced" or "expanded")
        prices_df: DataFrame with stock prices
        returns_df: DataFrame with stock returns
        **kwargs: Additional environment parameters
        
    Returns:
        Initialized portfolio environment
    """
    if env_type == "enhanced":
        return EnhancedPortfolioEnv(prices_df, returns_df, **kwargs)
    elif env_type == "expanded":
        return ExpandedEnhancedPortfolioEnv(prices_df, returns_df, **kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


if __name__ == "__main__":
    # Example usage
    print("Portfolio Environment Module")
    print("=" * 50)
    print("This module provides enhanced portfolio environments for RL training.")
    print("Use create_portfolio_env() to instantiate environments.")
