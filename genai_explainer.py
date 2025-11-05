# genai_explainer.py
import random

class AgentContext:
    def __init__(self, timestep, portfolio_weights, reward, episode, metrics):
        self.timestep = timestep
        self.portfolio_weights = portfolio_weights
        self.reward = reward
        self.episode = episode
        self.metrics = metrics


class Explanation:
    def __init__(self, reasoning, key_signals=None, risk_flags=None):
        self.reasoning = reasoning
        self.key_signals = key_signals or {}
        self.risk_flags = risk_flags or []


class GenAIExplainer:
    def __init__(self, model_backend="rule"):
        self.model_backend = model_backend

    def explain(self, agent_context, prices_df):
        """Simple placeholder explanation (can later use GPT)."""
        reasoning = (
            f"At timestep {agent_context.timestep}, the agent allocated "
            f"{agent_context.portfolio_weights} and achieved reward {agent_context.reward:.2f}. "
            "The policy suggests stable diversification with moderate tech exposure."
        )
        signals = {
            "momentum": round(random.uniform(0.2, 0.9), 2),
            "volatility": round(random.uniform(0.1, 0.5), 2)
        }
        risks = ["High concentration in tech sector", "Short-term volatility risk"]
        return Explanation(reasoning, signals, risks)
