# Smart Portfolio Allocator using Reinforcement Learning

A reinforcement learning approach to portfolio optimization and asset allocation for the capstone project.

## 🎯 Project Overview

This project implements a smart portfolio allocation system using reinforcement learning algorithms to optimize investment decisions across multiple assets. The system learns to rebalance portfolios dynamically based on market conditions and historical performance.

## 📁 Project Structure

```
Capstone_Portfolio_RL/
├── Files/                          # Stock market data (CSV files)
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Environment_Design.ipynb # RL Environment Design
│   ├── 03_Agent_Training.ipynb     # Model Training
│   └── 04_Results.ipynb            # Results & Visualization
├── src/                            # Core Python modules
│   ├── data/                       # Data loading & preprocessing
│   ├── environment/                # RL environment implementation
│   ├── agents/                     # RL agent implementations
│   ├── utils/                      # Utility functions
│   └── backtesting/                # Backtesting framework
├── scripts/                        # Training & evaluation scripts
├── models/                         # Saved model checkpoints
├── results/                        # Training logs & visualizations
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd Capstone_Portfolio_RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis

```bash
# Start Jupyter Lab
jupyter lab

# Open and run: notebooks/01_EDA.ipynb
```

### 3. Train RL Model

```bash
# Run training script
python scripts/train.py

# Evaluate trained model
python scripts/evaluate.py
```

## 📊 Features

- **Multi-asset Portfolio Management**: Support for multiple stocks/assets
- **RL Environment**: Custom Gym environment for portfolio optimization
- **Multiple Algorithms**: DQN, PPO, A3C, SAC implementations
- **Risk Management**: Sharpe ratio, volatility, drawdown considerations
- **Backtesting**: Historical performance evaluation
- **Visualization**: Comprehensive performance analysis and plotting

## 🧠 Reinforcement Learning Approach

### State Space
- Current portfolio weights
- Recent price changes and technical indicators
- Risk metrics and market conditions

### Action Space
- Portfolio weight adjustments
- Buy/Hold/Sell decisions per asset

### Reward Function
- Risk-adjusted returns (Sharpe ratio)
- Transaction cost penalties
- Drawdown penalties

## 📈 Data

The project uses stock market data stored in the `Files/` directory:
- Individual stock CSV files (AAPL, AMZN, GOOG, etc.)
- Historical price and volume data
- SP500 benchmark data

## 🔧 Dependencies

Key libraries used:
- **Data Science**: pandas, numpy, matplotlib, seaborn
- **Machine Learning**: scikit-learn, tensorflow, torch
- **Reinforcement Learning**: stable-baselines3, gymnasium, ray[rllib]
- **Financial Analysis**: yfinance, ta, quantlib
- **Development**: jupyter, jupyterlab

## 📝 Usage Examples

### Basic Data Loading
```python
from src.data.data_loader import StockDataLoader

loader = StockDataLoader()
data = loader.load_stock_data()
```

### Environment Setup
```python
from src.environment.portfolio_env import PortfolioEnv

env = PortfolioEnv(data, initial_capital=10000)
```

### Agent Training
```python
from src.agents.dqn_agent import DQNAgent

agent = DQNAgent(env)
agent.train(episodes=1000)
```

## 📊 Results & Performance

Results will be generated in the `results/` directory including:
- Training progress plots
- Portfolio performance comparisons
- Risk metrics analysis
- Backtesting results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of a capstone academic project.

## 👨‍💻 Author

**Subhrajeet** - Capstone Project

## 📚 References

- Reinforcement Learning for Portfolio Management
- Modern Portfolio Theory
- Risk Management in Financial Markets
Subhrajeet
Shubham
Jashraj