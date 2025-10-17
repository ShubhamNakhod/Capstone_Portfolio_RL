"""
OpenAI Data Integration Module

This module integrates your existing stock market data with OpenAI's API,
providing seamless data preparation and formatting for OpenAI-powered
trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class OpenAIDataIntegrator:
    """
    Integrates stock market data with OpenAI API requirements.
    
    This class handles data loading, preprocessing, and formatting
    to make your existing stock data work seamlessly with OpenAI's
    natural language processing capabilities.
    """
    
    def __init__(self, data_directory: str = "../Files"):
        """
        Initialize the data integrator.
        
        Args:
            data_directory: Path to directory containing stock data files
        """
        self.data_directory = Path(data_directory)
        self.stock_data = {}
        self.combined_prices = None
        self.daily_returns = None
        self.technical_indicators = {}
        
    def load_all_stock_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all stock data from CSV files.
        
        Returns:
            Dictionary with stock symbols as keys and DataFrames as values
        """
        csv_files = list(self.data_directory.glob("*.csv"))
        
        for file_path in csv_files:
            stock_name = file_path.stem
            
            # Skip non-stock files
            if stock_name in ['Annexure-I']:
                continue
                
            try:
                df = self._load_single_stock(file_path)
                if df is not None and 'close' in df.columns:
                    self.stock_data[stock_name] = df
                    print(f"✓ Loaded {stock_name}: {len(df)} rows")
            except Exception as e:
                print(f"✗ Failed to load {stock_name}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.stock_data)} stocks")
        return self.stock_data
    
    def _load_single_stock(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a single stock CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with stock data or None if loading fails
        """
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'Date': 'date', 'DATE': 'date', 'Date/Time': 'date',
                'Open': 'open', 'OPEN': 'open',
                'High': 'high', 'HIGH': 'high',
                'Low': 'low', 'LOW': 'low',
                'Close': 'close', 'CLOSE': 'close',
                'Adj Close': 'adj_close', 'Adj_Close': 'adj_close',
                'Volume': 'volume', 'VOLUME': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
                df = df.set_index('date')
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def create_combined_dataset(self, price_column: str = 'close') -> pd.DataFrame:
        """
        Create combined dataset with all stock prices.
        
        Args:
            price_column: Column to use for prices ('close' or 'adj_close')
            
        Returns:
            Combined DataFrame with all stock prices
        """
        combined_data = {}
        
        for stock_name, df in self.stock_data.items():
            if price_column in df.columns:
                # Remove duplicates and handle missing values
                price_series = df[price_column].dropna()
                if not price_series.empty:
                    combined_data[stock_name] = price_series
        
        if combined_data:
            self.combined_prices = pd.DataFrame(combined_data)
            
            # Align all series to common date range
            self.combined_prices = self.combined_prices.dropna()
            
            print(f"Combined dataset shape: {self.combined_prices.shape}")
            print(f"Date range: {self.combined_prices.index.min()} to {self.combined_prices.index.max()}")
            
            return self.combined_prices
        else:
            raise ValueError("No valid price data found")
    
    def calculate_returns(self, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame with returns
        """
        if self.combined_prices is None:
            raise ValueError("Must create combined dataset first")
        
        if method == 'simple':
            self.daily_returns = self.combined_prices.pct_change().dropna()
        elif method == 'log':
            self.daily_returns = np.log(self.combined_prices / self.combined_prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        print(f"Returns dataset shape: {self.daily_returns.shape}")
        return self.daily_returns
    
    def calculate_technical_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate technical indicators for all stocks.
        
        Returns:
            Dictionary with technical indicators for each stock
        """
        if self.combined_prices is None:
            raise ValueError("Must create combined dataset first")
        
        self.technical_indicators = {}
        
        for symbol in self.combined_prices.columns:
            prices = self.combined_prices[symbol]
            returns = self.daily_returns[symbol] if self.daily_returns is not None else prices.pct_change()
            
            # Calculate technical indicators
            indicators = pd.DataFrame(index=prices.index)
            
            # Moving averages
            indicators[f'{symbol}_MA_5'] = prices.rolling(5).mean()
            indicators[f'{symbol}_MA_20'] = prices.rolling(20).mean()
            indicators[f'{symbol}_MA_50'] = prices.rolling(50).mean()
            
            # Price ratios
            indicators[f'{symbol}_price_ratio_5'] = prices / indicators[f'{symbol}_MA_5']
            indicators[f'{symbol}_price_ratio_20'] = prices / indicators[f'{symbol}_MA_20']
            
            # Volatility
            indicators[f'{symbol}_volatility_5'] = returns.rolling(5).std()
            indicators[f'{symbol}_volatility_20'] = returns.rolling(20).std()
            
            # Momentum
            indicators[f'{symbol}_momentum_5'] = returns.rolling(5).sum()
            indicators[f'{symbol}_momentum_20'] = returns.rolling(20).sum()
            
            # RSI approximation
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gain = gains.rolling(14).mean()
            avg_loss = losses.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-8)
            indicators[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            ma_20 = indicators[f'{symbol}_MA_20']
            std_20 = prices.rolling(20).std()
            indicators[f'{symbol}_bb_upper'] = ma_20 + (2 * std_20)
            indicators[f'{symbol}_bb_lower'] = ma_20 - (2 * std_20)
            indicators[f'{symbol}_bb_position'] = (prices - indicators[f'{symbol}_bb_lower']) / (indicators[f'{symbol}_bb_upper'] - indicators[f'{symbol}_bb_lower'])
            
            self.technical_indicators[symbol] = indicators.dropna()
        
        print(f"Calculated technical indicators for {len(self.technical_indicators)} stocks")
        return self.technical_indicators
    
    def format_for_openai(self, 
                         date: str = None, 
                         lookback_days: int = 5) -> Dict:
        """
        Format market data for OpenAI consumption.
        
        Args:
            date: Specific date to format (default: latest available)
            lookback_days: Number of days to include in lookback
            
        Returns:
            Dictionary formatted for OpenAI API
        """
        if self.combined_prices is None:
            raise ValueError("Must create combined dataset first")
        
        # Select date
        if date is None:
            target_date = self.combined_prices.index[-1]
        else:
            target_date = pd.to_datetime(date)
        
        # Get data around target date
        date_idx = self.combined_prices.index.get_loc(target_date, method='nearest')
        start_idx = max(0, date_idx - lookback_days)
        end_idx = min(len(self.combined_prices), date_idx + 1)
        
        # Extract data
        price_data = self.combined_prices.iloc[start_idx:end_idx]
        return_data = self.daily_returns.iloc[start_idx:end_idx] if self.daily_returns is not None else None
        
        # Format for OpenAI
        openai_data = {
            "date": target_date.strftime("%Y-%m-%d"),
            "stocks": {},
            "market_summary": {},
            "technical_analysis": {}
        }
        
        # Current prices and recent performance
        current_prices = price_data.iloc[-1].to_dict()
        
        for symbol in self.combined_prices.columns:
            stock_info = {
                "current_price": current_prices[symbol],
                "price_history": price_data[symbol].tolist(),
                "dates": [d.strftime("%Y-%m-%d") for d in price_data.index]
            }
            
            # Add returns if available
            if return_data is not None and symbol in return_data.columns:
                recent_returns = return_data[symbol].tolist()
                stock_info["returns"] = recent_returns
                stock_info["avg_return_5d"] = np.mean(recent_returns[-5:]) if len(recent_returns) >= 5 else 0
                stock_info["volatility_5d"] = np.std(recent_returns[-5:]) if len(recent_returns) >= 5 else 0
            
            # Add technical indicators if available
            if symbol in self.technical_indicators:
                tech_data = self.technical_indicators[symbol]
                if target_date in tech_data.index:
                    tech_row = tech_data.loc[target_date]
                    stock_info["technical_indicators"] = {
                        "ma_5": tech_row.get(f'{symbol}_MA_5', None),
                        "ma_20": tech_row.get(f'{symbol}_MA_20', None),
                        "price_ratio_20": tech_row.get(f'{symbol}_price_ratio_20', None),
                        "rsi": tech_row.get(f'{symbol}_rsi', None),
                        "bb_position": tech_row.get(f'{symbol}_bb_position', None)
                    }
            
            openai_data["stocks"][symbol] = stock_info
        
        # Market summary
        if return_data is not None:
            latest_returns = return_data.iloc[-1].to_dict() if len(return_data) > 0 else {}
            positive_stocks = sum(1 for ret in latest_returns.values() if ret > 0)
            total_stocks = len(latest_returns)
            
            openai_data["market_summary"] = {
                "total_stocks": total_stocks,
                "positive_stocks": positive_stocks,
                "market_sentiment": "Bullish" if positive_stocks > total_stocks * 0.6 else "Bearish" if positive_stocks < total_stocks * 0.4 else "Neutral",
                "avg_return": np.mean(list(latest_returns.values())) if latest_returns else 0,
                "market_volatility": np.std(list(latest_returns.values())) if latest_returns else 0
            }
        
        return openai_data
    
    def create_natural_language_summary(self, date: str = None) -> str:
        """
        Create natural language summary of market data for OpenAI.
        
        Args:
            date: Specific date to summarize (default: latest)
            
        Returns:
            Natural language summary string
        """
        data = self.format_for_openai(date)
        
        summary = f"""
        === MARKET ANALYSIS FOR {data['date']} ===
        
        MARKET OVERVIEW:
        • Total Stocks: {data['market_summary']['total_stocks']}
        • Positive Stocks: {data['market_summary']['positive_stocks']}
        • Market Sentiment: {data['market_summary']['market_sentiment']}
        • Average Return: {data['market_summary']['avg_return']:.2%}
        • Market Volatility: {data['market_summary']['market_volatility']:.2%}
        
        INDIVIDUAL STOCK ANALYSIS:
        """
        
        for symbol, info in data['stocks'].items():
            summary += f"\n{symbol}:\n"
            summary += f"  • Current Price: ${info['current_price']:.2f}\n"
            
            if 'avg_return_5d' in info:
                summary += f"  • 5-Day Avg Return: {info['avg_return_5d']:.2%}\n"
                summary += f"  • 5-Day Volatility: {info['volatility_5d']:.2%}\n"
            
            if 'technical_indicators' in info:
                tech = info['technical_indicators']
                if tech['rsi'] is not None:
                    rsi_signal = "Overbought" if tech['rsi'] > 70 else "Oversold" if tech['rsi'] < 30 else "Neutral"
                    summary += f"  • RSI: {tech['rsi']:.1f} ({rsi_signal})\n"
                
                if tech['price_ratio_20'] is not None:
                    ma_signal = "Above MA" if tech['price_ratio_20'] > 1 else "Below MA"
                    summary += f"  • Price vs 20-Day MA: {tech['price_ratio_20']:.3f} ({ma_signal})\n"
        
        return summary
    
    def get_portfolio_data_for_openai(self, 
                                    portfolio_weights: Dict[str, float],
                                    date: str = None) -> str:
        """
        Get portfolio data formatted for OpenAI analysis.
        
        Args:
            portfolio_weights: Dictionary with stock symbols and weights
            date: Specific date (default: latest)
            
        Returns:
            Formatted portfolio data string
        """
        data = self.format_for_openai(date)
        
        portfolio_summary = f"""
        === PORTFOLIO ANALYSIS FOR {data['date']} ===
        
        CURRENT ALLOCATION:
        """
        
        total_value = 10000  # Placeholder for portfolio value
        
        for symbol, weight in portfolio_weights.items():
            if symbol in data['stocks']:
                stock_info = data['stocks'][symbol]
                position_value = total_value * weight
                
                portfolio_summary += f"\n{symbol}:\n"
                portfolio_summary += f"  • Allocation: {weight:.1%}\n"
                portfolio_summary += f"  • Position Value: ${position_value:.2f}\n"
                portfolio_summary += f"  • Current Price: ${stock_info['current_price']:.2f}\n"
                
                if 'avg_return_5d' in stock_info:
                    portfolio_summary += f"  • 5-Day Return: {stock_info['avg_return_5d']:.2%}\n"
        
        # Portfolio-level analysis
        portfolio_return = 0
        portfolio_volatility = 0
        
        for symbol, weight in portfolio_weights.items():
            if symbol in data['stocks'] and 'avg_return_5d' in data['stocks'][symbol]:
                stock_return = data['stocks'][symbol]['avg_return_5d']
                stock_vol = data['stocks'][symbol]['volatility_5d']
                
                portfolio_return += weight * stock_return
                portfolio_volatility += (weight ** 2) * (stock_vol ** 2)
        
        portfolio_volatility = np.sqrt(portfolio_volatility)
        
        portfolio_summary += f"""
        
        PORTFOLIO METRICS:
        • Expected Return (5-day): {portfolio_return:.2%}
        • Portfolio Volatility: {portfolio_volatility:.2%}
        • Risk-Adjusted Return: {portfolio_return/portfolio_volatility:.3f} (if vol > 0)
        
        MARKET CONTEXT:
        • Market Sentiment: {data['market_summary']['market_sentiment']}
        • Market Volatility: {data['market_summary']['market_volatility']:.2%}
        """
        
        return portfolio_summary
    
    def create_training_data_for_fine_tuning(self, 
                                           episode_length: int = 252,
                                           num_episodes: int = 100) -> List[Dict]:
        """
        Create training data for OpenAI fine-tuning from historical data.
        
        Args:
            episode_length: Length of each training episode
            num_episodes: Number of episodes to generate
            
        Returns:
            List of training examples for fine-tuning
        """
        if self.daily_returns is None:
            raise ValueError("Must calculate returns first")
        
        training_data = []
        
        for episode in range(num_episodes):
            # Random start date
            start_idx = np.random.randint(0, len(self.daily_returns) - episode_length)
            
            for step in range(start_idx, start_idx + episode_length - 1):
                if step >= len(self.daily_returns) - 1:
                    break
                
                # Get market state
                market_data = self.format_for_openai(self.daily_returns.index[step].strftime("%Y-%m-%d"))
                
                # Calculate optimal action based on next day's returns
                next_returns = self.daily_returns.iloc[step + 1]
                best_stock = next_returns.idxmax()
                worst_stock = next_returns.idxmin()
                
                # Create training example
                user_message = f"""
                Market data for {market_data['date']}:
                {self.create_natural_language_summary(market_data['date'])}
                
                What trading action should I take?
                """
                
                assistant_message = f"""
                Based on the market analysis, I recommend:
                
                1. **Buy {best_stock}**: Shows highest expected return for tomorrow ({next_returns[best_stock]:.2%})
                2. **Consider selling {worst_stock}**: Shows lowest expected return ({next_returns[worst_stock]:.2%})
                3. **Monitor market sentiment**: Currently {market_data['market_summary']['market_sentiment'].lower()}
                
                This recommendation is based on the technical indicators and recent price movements.
                """
                
                training_data.append({
                    "messages": [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_message}
                    ]
                })
        
        print(f"Created {len(training_data)} training examples")
        return training_data
    
    def save_data_for_openai(self, output_file: str = "openai_market_data.json"):
        """
        Save formatted market data for OpenAI usage.
        
        Args:
            output_file: Output file path
        """
        if self.combined_prices is None:
            raise ValueError("Must create combined dataset first")
        
        # Get latest data
        latest_data = self.format_for_openai()
        
        with open(output_file, 'w') as f:
            json.dump(latest_data, f, indent=2, default=str)
        
        print(f"Saved market data to {output_file}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize data integrator
    integrator = OpenAIDataIntegrator()
    
    # Load stock data
    stock_data = integrator.load_all_stock_data()
    
    # Create combined dataset
    combined_prices = integrator.create_combined_dataset()
    
    # Calculate returns
    daily_returns = integrator.calculate_returns()
    
    # Calculate technical indicators
    technical_indicators = integrator.calculate_technical_indicators()
    
    # Format for OpenAI
    openai_data = integrator.format_for_openai()
    print("OpenAI formatted data keys:", list(openai_data.keys()))
    
    # Create natural language summary
    summary = integrator.create_natural_language_summary()
    print("\nNatural Language Summary:")
    print(summary[:500] + "..." if len(summary) > 500 else summary)
    
    # Create portfolio analysis
    sample_portfolio = {
        'AAPL': 0.3,
        'AMZN': 0.25,
        'GOOG': 0.2,
        'MSFT': 0.15,
        'JNJ': 0.1
    }
    
    portfolio_analysis = integrator.get_portfolio_data_for_openai(sample_portfolio)
    print("\nPortfolio Analysis:")
    print(portfolio_analysis[:500] + "..." if len(portfolio_analysis) > 500 else portfolio_analysis)
    
    # Save data
    integrator.save_data_for_openai()
    
    print("\n✓ OpenAI data integration completed successfully!")
