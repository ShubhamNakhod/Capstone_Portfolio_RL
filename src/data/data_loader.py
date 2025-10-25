"""
Data Loading and Preprocessing Module

This module handles loading, cleaning, and preprocessing of stock market data
for the portfolio optimization environment. Based on the EDA analysis in
notebooks/01_EDA.ipynb.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


def load_stock_data(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load stock data from CSV file and standardize format.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with standardized column names and datetime index, or None if error
    """
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names (common variations)
        column_mapping = {
            'Date': 'date',
            'DATE': 'date',
            'Date/Time': 'date',
            'Open': 'open',
            'OPEN': 'open',
            'High': 'high', 
            'HIGH': 'high',
            'Low': 'low',
            'LOW': 'low',
            'Close': 'close',
            'CLOSE': 'close',
            'Adj Close': 'adj_close',
            'Adj_Close': 'adj_close',
            'Volume': 'volume',
            'VOLUME': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert date column with dayfirst=True to handle DD-MM-YYYY format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
            df = df.set_index('date')
        
        return df
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_all_stock_data(data_dir: Union[str, Path], 
                        selected_stocks: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all stock data from a directory.
    
    Args:
        data_dir: Directory containing CSV files
        selected_stocks: Optional list of specific stocks to load
        
    Returns:
        Dictionary mapping stock names to DataFrames
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    stock_data = {}
    failed_files = []
    
    for file_path in csv_files:
        stock_name = file_path.stem
        
        # Skip if specific stocks are requested and this isn't one of them
        if selected_stocks and stock_name not in selected_stocks:
            continue
            
        df = load_stock_data(file_path)
        
        if df is not None:
            stock_data[stock_name] = df
            print(f"✓ Loaded {stock_name}: {len(df)} rows, columns: {list(df.columns)}")
        else:
            failed_files.append(stock_name)
    
    if failed_files:
        print(f"Failed to load: {failed_files}")
    
    return stock_data


def create_combined_dataset(stock_data: Dict[str, pd.DataFrame], 
                          price_column: str = 'close') -> Optional[pd.DataFrame]:
    """
    Create a combined dataset with all stock prices.
    
    Args:
        stock_data: Dictionary mapping stock names to DataFrames
        price_column: Column name to use for prices
        
    Returns:
        Combined DataFrame with all stock prices, or None if no valid data
    """
    combined_data = {}
    
    for stock_name, df in stock_data.items():
        if price_column in df.columns:
            # Remove duplicate dates, keeping the first occurrence
            if df.index.duplicated().any():
                print(f"⚠ Warning: {stock_name} has {df.index.duplicated().sum()} duplicate dates. Removing duplicates...")
                df = df[~df.index.duplicated(keep='first')]
            
            # Only include stocks with valid price data (not NaN)
            price_series = df[price_column]
            if not price_series.isna().all():  # Only include if there's at least some valid data
                combined_data[stock_name] = price_series
    
    if combined_data:
        # Use pd.concat instead of pd.DataFrame to handle different indices better
        combined_df = pd.concat(combined_data, axis=1)
        combined_df.columns = list(combined_data.keys())
        return combined_df
    else:
        return None


def calculate_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: DataFrame with price data
        method: Method to calculate returns ('simple' or 'log')
        
    Returns:
        DataFrame with returns
    """
    if method == 'simple':
        returns = prices.pct_change().dropna()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'simple' or 'log'")
    return returns


def create_diversified_stock_selection() -> List[str]:
    """
    Create a diversified stock selection across sectors for better portfolio diversification.
    Based on EDA analysis and sector categorization.
    
    Returns:
        List of selected stock symbols
    """
    # Sector-based stock selection for diversification
    diversified_stocks = {
        'Technology': ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'FB', 'IBM'],
        'Financial': ['GS', 'MS', 'WFC', 'BCS'],
        'Healthcare': ['JNJ', 'MRK', 'PFE', 'UNH'],
        'Airlines': ['AAL', 'DAL', 'LUV'],
        'Index': ['SP500']
    }
    
    # Select 2-3 stocks from each sector for balanced diversification
    selected_stocks = []
    for sector, stocks in diversified_stocks.items():
        if sector == 'Index':
            selected_stocks.extend(stocks)  # Include all index funds
        else:
            # Select top 2-3 stocks from each sector
            selected_stocks.extend(stocks[:3])
    
    return selected_stocks


def create_simple_stock_selection() -> List[str]:
    """
    Create a simple stock selection for prototyping (6 stocks).
    
    Returns:
        List of selected stock symbols
    """
    return ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'JNJ', 'SP500']


def validate_data_quality(prices_df: pd.DataFrame, 
                         returns_df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data quality and return summary statistics.
    
    Args:
        prices_df: DataFrame with price data
        returns_df: DataFrame with returns data
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_metrics = {
        'n_stocks': len(prices_df.columns),
        'n_days': len(prices_df),
        'date_range': (prices_df.index.min(), prices_df.index.max()),
        'missing_values_prices': prices_df.isnull().sum().sum(),
        'missing_values_returns': returns_df.isnull().sum().sum(),
        'extreme_returns': 0,
        'data_quality_score': 0.0
    }
    
    # Check for extreme returns (potential data issues)
    extreme_threshold = 0.5  # 50% daily return
    extreme_returns = returns_df[(returns_df > extreme_threshold) | (returns_df < -extreme_threshold)]
    quality_metrics['extreme_returns'] = len(extreme_returns)
    
    # Calculate data quality score (0-1, higher is better)
    missing_penalty = (quality_metrics['missing_values_prices'] + quality_metrics['missing_values_returns']) / (len(prices_df) * len(prices_df.columns) * 2)
    extreme_penalty = quality_metrics['extreme_returns'] / (len(returns_df) * len(returns_df.columns))
    
    quality_metrics['data_quality_score'] = max(0, 1 - missing_penalty - extreme_penalty)
    
    return quality_metrics


def prepare_training_data(prices_df: pd.DataFrame, 
                         returns_df: pd.DataFrame,
                         train_split: float = 0.7,
                         val_split: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for training with time-series splits.
    
    Args:
        prices_df: DataFrame with price data
        returns_df: DataFrame with returns data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        
    Returns:
        Tuple of (train_prices, val_prices, test_prices, train_returns, val_returns, test_returns)
    """
    n_days = len(prices_df)
    train_end = int(n_days * train_split)
    val_end = int(n_days * (train_split + val_split))
    
    # Split prices
    train_prices = prices_df.iloc[:train_end]
    val_prices = prices_df.iloc[train_end:val_end]
    test_prices = prices_df.iloc[val_end:]
    
    # Split returns
    train_returns = returns_df.iloc[:train_end]
    val_returns = returns_df.iloc[train_end:val_end]
    test_returns = returns_df.iloc[val_end:]
    
    return train_prices, val_prices, test_prices, train_returns, val_returns, test_returns


def load_and_prepare_data(data_dir: Union[str, Path],
                         stock_selection: str = "diversified",
                         **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """
    Complete data loading and preparation pipeline.
    
    Args:
        data_dir: Directory containing CSV files
        stock_selection: Type of stock selection ("diversified", "simple", or "all")
        **kwargs: Additional parameters for data preparation
        
    Returns:
        Tuple of (prices_df, returns_df, quality_metrics)
    """
    # Select stocks based on type
    if stock_selection == "diversified":
        selected_stocks = create_diversified_stock_selection()
    elif stock_selection == "simple":
        selected_stocks = create_simple_stock_selection()
    elif stock_selection == "all":
        selected_stocks = None
    else:
        raise ValueError("stock_selection must be 'diversified', 'simple', or 'all'")
    
    print(f"Loading data with {stock_selection} stock selection...")
    if selected_stocks:
        print(f"Selected stocks: {selected_stocks}")
    
    # Load stock data
    stock_data = load_all_stock_data(data_dir, selected_stocks)
    
    if not stock_data:
        raise ValueError("No valid stock data loaded")
    
    # Create combined dataset
    prices_df = create_combined_dataset(stock_data, price_column='close')
    if prices_df is None:
        raise ValueError("Failed to create combined price dataset")
    
    # Calculate returns
    returns_df = calculate_returns(prices_df, method='simple')
    
    # Validate data quality
    quality_metrics = validate_data_quality(prices_df, returns_df)
    
    print(f"\nData Summary:")
    print(f"  Stocks: {quality_metrics['n_stocks']}")
    print(f"  Days: {quality_metrics['n_days']}")
    print(f"  Date range: {quality_metrics['date_range'][0]} to {quality_metrics['date_range'][1]}")
    print(f"  Data quality score: {quality_metrics['data_quality_score']:.3f}")
    print(f"  Missing values: {quality_metrics['missing_values_prices']} (prices), {quality_metrics['missing_values_returns']} (returns)")
    print(f"  Extreme returns: {quality_metrics['extreme_returns']}")
    
    return prices_df, returns_df, quality_metrics


if __name__ == "__main__":
    # Example usage
    print("Data Loading Module")
    print("=" * 50)
    print("This module handles loading and preprocessing of stock market data.")
    print("Use load_and_prepare_data() for complete data preparation pipeline.")
    
    # Example with sample data
    try:
        data_dir = Path("../Files")
        if data_dir.exists():
            prices, returns, quality = load_and_prepare_data(data_dir, stock_selection="simple")
            print(f"\nExample: Loaded {len(prices.columns)} stocks with {len(prices)} days of data")
        else:
            print("Data directory not found. Please ensure Files/ directory exists.")
    except Exception as e:
        print(f"Example failed: {e}")
