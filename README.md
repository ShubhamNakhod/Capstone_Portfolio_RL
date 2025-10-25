# Smart Portfolio Allocator - RL Portfolio Optimization

A comprehensive reinforcement learning system for portfolio optimization using advanced RL algorithms and market data analysis.

## 🎯 Project Overview

This project implements a sophisticated portfolio optimization system using reinforcement learning algorithms (PPO, DQN) to automatically allocate assets based on market conditions, technical indicators, and risk metrics. The system incorporates insights from comprehensive EDA analysis and provides both discrete and continuous action spaces for different trading strategies.

## 📁 Project Structure

```
Capstone_Portfolio_RL/
├── Files/                          # Stock market data (CSV files)
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Environment_Design.ipynb # RL Environment Design & Prototyping
│   └── 03_Agent_Training.ipynb    # Agent Training & Experimentation
├── src/                            # Core Python modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py         # Data loading and preprocessing
│   ├── environment/
│   │   ├── __init__.py
│   │   └── portfolio_env.py       # RL environments (Enhanced & Expanded)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Base agent class and utilities
│   │   ├── ppo_agent.py          # PPO agent implementation
│   │   └── dqn_agent.py          # DQN agent implementation
│   └── utils/
│       └── __init__.py
├── scripts/                        # Training and evaluation scripts
│   ├── train.py                   # Main training script
│   └── evaluate.py                # Evaluation and backtesting script
├── models/                         # Saved trained models
├── results/                        # Training logs, plots, and results
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Capstone_Portfolio_RL

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure your stock data is in the `Files/` directory as CSV files with columns: Date, Open, High, Low, Close, Volume.

### 3. Training an Agent

```bash
# Train PPO agent on enhanced environment with diversified stocks
python scripts/train.py --agent ppo --env enhanced --stocks diversified --episodes 1000

# Train DQN agent on simple environment
python scripts/train.py --agent dqn --env enhanced --stocks simple --episodes 1000
```

### 4. Evaluating a Trained Agent

```bash
# Evaluate trained PPO agent
python scripts/evaluate.py --agent ppo --model_path models/final_model.pth --episodes 100 --compare

# Evaluate with benchmark comparison
python scripts/evaluate.py --agent dqn --model_path models/dqn_agent.pth --episodes 100 --compare
```

## 🧠 Key Features

### Enhanced RL Environment
- **State Space**: ~251 features including:
  - Portfolio weights (16 stocks)
  - Technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands)
  - Correlation features for diversification
  - Market regime indicators (volatility, trend, cycle)
  - Risk metrics (VaR, drawdown)

- **Action Space**: Continuous softmax-normalized portfolio weights
- **Reward Function**: Multi-component reward incorporating:
  - Sharpe ratio with risk-free rate
  - VaR penalties
  - Drawdown penalties
  - Correlation penalties
  - Market regime bonuses

### RL Agents

#### PPO Agent (Proximal Policy Optimization)
- Continuous action space with softmax normalization
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective
- Multiple PPO epochs per update
- **Best for**: Continuous portfolio weight optimization

#### DQN Agent (Deep Q-Network)
- Discrete action space with predefined strategies
- Experience replay buffer
- Target network updates
- Epsilon-greedy exploration
- **Best for**: Discrete portfolio allocation strategies

### Data Integration
- **25+ stocks** across multiple sectors
- **10+ years** of historical data (2010-2020)
- **Automated preprocessing** with quality validation
- **Multiple stock selections**: Simple (6 stocks), Diversified (13 stocks), All (25+ stocks)

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total return, annualized return, excess return
- **Risk**: Volatility, maximum drawdown, VaR (Value at Risk)
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio
- **Diversification**: Herfindahl-Hirschman Index, effective number of stocks/sectors
- **Trading**: Transaction costs, turnover, win rate

## 🔬 Research Insights

Based on comprehensive EDA analysis:

1. **Correlation Analysis**: 15 highly correlated pairs identified (avg correlation: 0.410)
2. **Risk Metrics**: VaR, max drawdown, and Sharpe ratios calculated for all stocks
3. **Diversification**: Moderate diversification potential across sectors
4. **Market Regimes**: Volatility and trend regimes detected and incorporated
5. **Technical Indicators**: Complete suite implemented based on EDA recommendations

## 🎛️ Configuration Options

### Environment Types
- **Enhanced**: 6 stocks with full technical indicators (~66 features)
- **Expanded**: 13 diversified stocks across sectors (~185 features)

### Stock Selections
- **Simple**: AAPL, AMZN, GOOG, MSFT, JNJ, SP500
- **Diversified**: Multi-sector selection (Tech, Financial, Healthcare, Airlines, Index)
- **All**: Complete dataset (25+ stocks)

### Training Parameters
- **Episode Length**: 252 days (1 trading year)
- **Transaction Costs**: 0.1% per trade
- **Lookback Window**: 20 days
- **Risk-Free Rate**: 2% (from EDA analysis)

## 📈 Usage Examples

### Training with Custom Parameters

```python
from src.data.data_loader import load_and_prepare_data
from src.environment.portfolio_env import create_portfolio_env
from src.agents.ppo_agent import PPOAgent

# Load data
prices, returns, quality = load_and_prepare_data('../Files', stock_selection='diversified')

# Create environment
env = create_portfolio_env('expanded', prices, returns, episode_length=252)

# Create agent
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    learning_rate=3e-4,
    gamma=0.99,
    clip_ratio=0.2
)

# Training loop
for episode in range(1000):
    state, info = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done, truncated, info = env.step(action)
        agent.store_reward(reward, done or truncated)
        state = next_state
    agent.update()
```

### Evaluation and Backtesting

```python
from scripts.evaluate import run_backtest, compare_with_benchmarks

# Run backtest
backtest_results = run_backtest(env, agent, n_episodes=100)

# Compare with benchmarks
comparison = compare_with_benchmarks(env, agent, n_episodes=100)
```

## 🧪 Experimentation

The project includes comprehensive experimentation capabilities:

1. **Hyperparameter Tuning**: Learning rates, network architectures, reward functions
2. **Agent Comparison**: PPO vs DQN vs benchmark strategies
3. **Environment Analysis**: Simple vs expanded environments
4. **Stock Universe**: Different stock selections and sector allocations
5. **Training Dynamics**: Convergence analysis, stability metrics

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- Stable-Baselines3
- Pandas, NumPy, Matplotlib, Seaborn
- Gymnasium (OpenAI Gym)

See `requirements.txt` for complete dependency list.

## 🔧 Advanced Usage

### Custom Reward Functions

```python
class CustomPortfolioEnv(EnhancedPortfolioEnv):
    def _calculate_reward(self):
        # Implement custom reward logic
        portfolio_return = self.return_history[-1]
        # Add your custom reward components
        return custom_reward
```

### Custom Action Spaces

```python
class CustomDQNAgent(DQNAgent):
    def _create_action_strategies(self):
        # Define custom portfolio strategies
        strategies = [
            # Your custom strategies here
        ]
        return strategies
```

## 📊 Results and Visualization

The system provides comprehensive visualization:

- **Training Curves**: Episode rewards, losses, convergence
- **Performance Metrics**: Returns, risk, Sharpe ratios
- **Portfolio Analysis**: Weight distributions, sector allocations
- **Benchmark Comparison**: Against equal-weight, buy-hold, random strategies
- **Risk Analysis**: Drawdowns, volatility, correlation heatmaps

## 🚀 Next Steps

1. **Expand Stock Universe**: Add more stocks and sectors
2. **Advanced Algorithms**: Implement SAC, TD3, or ensemble methods
3. **Online Learning**: Real-time adaptation to market conditions
4. **Risk Management**: Advanced position sizing and risk controls
5. **Live Trading**: Integration with trading platforms (with proper safeguards)

## ⚠️ Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough testing and risk assessment before using any trading strategies with real money.

## 📚 References

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236)
- [Portfolio Optimization with RL](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3389758)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.