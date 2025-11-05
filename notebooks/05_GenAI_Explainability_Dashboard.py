# 05_GenAI_Explainability_Dashboard.py
# -------------------------------------------------------------------------
# Streamlit dashboard for a hybrid GenAI + RL investing assistant.
# Combines local RL agent intelligence with real-time market data,
# human-like reasoning, and explainable recommendations.
# -------------------------------------------------------------------------

from __future__ import annotations
import os, math, textwrap, re
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Internet data
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

# Local imports
_dl_ok = False
_agents_ok = False
try:
    from data_loader import load_and_prepare_data
    _dl_ok = True
except Exception:
    try:
        from src.data.data_loader import load_and_prepare_data
        _dl_ok = True
    except Exception:
        pass

try:
    from ppo_agent import PPOAgent
    from dqn_agent import DQNAgent
    _agents_ok = True
except Exception:
    try:
        from src.agents.ppo_agent import PPOAgent
        from src.agents.dqn_agent import DQNAgent
        _agents_ok = True
    except Exception:
        pass

# OpenAI setup
_openai_ready = False
_openai_mode = None
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        _openai_ready = True
        _openai_mode = "client_v1"
except Exception:
    try:
        import openai
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            _openai_ready = True
            _openai_mode = "legacy"
    except Exception:
        _openai_ready = False

# Streamlit UI
st.set_page_config(page_title="GenAI Explainability Dashboard", layout="wide")

with st.sidebar:
    st.title("GenAI + RL Trading Assistant")
    agent_choice = st.selectbox("Select RL Agent", ["PPO", "DQN"], index=0)
    data_mode = st.selectbox("Data Source", ["Local CSVs", "Internet (yfinance)"], index=0)
    data_dir = st.text_input("Local data directory", value=str(Path.cwd()))
    ppo_ckpt = st.text_input("PPO checkpoint", value="ppo_agent.pth")
    dqn_ckpt = st.text_input("DQN checkpoint", value="dqn_agent.pth")

    st.divider()
    st.text("Investment Profile")
    init_capital = st.number_input("Capital (USD)", min_value=1000, value=10000, step=1000)
    risk_tolerance = st.select_slider("Risk tolerance", ["low", "medium", "high"], value="medium")
    horizon_years = st.slider("Investment horizon (years)", 1, 10, 3)
    need_income = st.checkbox("I need regular income (dividends)", value=False)
    st.caption("Tabs below: Chat | Data Explorer | RL Insights")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prices_df" not in st.session_state:
    st.session_state.prices_df = None
if "returns_df" not in st.session_state:
    st.session_state.returns_df = None

# Helper functions ----------------------------------------------------------
def _human_reply(text: str) -> str:
    t = text.strip().lower()
    if any(x in t for x in ["hello", "hi", "hey", "yo"]):
        return "Hello. How can I help with markets or your portfolio today?"
    if t.endswith("?"):
        return "Good question. Let’s reason through it using your market data."
    return "Got it. Let’s analyze this step by step."

def fetch_internet_prices(tickers: List[str], period: str = "6mo") -> Optional[pd.DataFrame]:
    if not YF_OK:
        st.warning("yfinance not installed.")
        return None
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if "Close" in data.columns:
            close = data["Close"].copy()
            close.columns = [c if isinstance(c, str) else c[1] for c in close.columns]
            return close.dropna(how="all")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    return None

def suggest_allocation(returns_df: pd.DataFrame, risk: str, need_income: bool, capital: float):
    if returns_df is None or returns_df.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    mu = returns_df.mean() * 252
    cov = returns_df.cov() * 252
    target_vol = {"low": 0.10, "medium": 0.18, "high": 0.28}[risk]
    cov_reg = cov + 0.5 * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov_reg.values)
    raw = np.maximum(inv @ mu.values, 0)
    w = raw / raw.sum()
    vol = math.sqrt(float(w @ cov.values @ w))
    w = w * min(1.0, target_vol / (vol + 1e-9))
    w = w / w.sum()
    weights = pd.Series(w, index=returns_df.columns)
    if need_income:
        for i in weights.index:
            if any(x in i for x in ["JNJ", "KO", "PG", "UNH"]):
                weights[i] *= 1.15
        weights /= weights.sum()
    alloc = (weights * capital).round(2)
    table = pd.DataFrame({"Weight %": (weights * 100).round(2), "Allocation ($)": alloc})
    return weights, table

def run_backtest(prices_df: pd.DataFrame, weights: pd.Series):
    if prices_df is None or weights is None or weights.empty:
        return pd.DataFrame()
    common = [c for c in weights.index if c in prices_df.columns]
    rets = prices_df[common].pct_change().dropna()
    port = (rets @ weights[common]).to_frame("Return")
    port["Cumulative"] = (1 + port["Return"]).cumprod()
    return port

# Tabs ----------------------------------------------------------------------
chat_tab, data_tab, rl_tab = st.tabs(["Chat", "Data Explorer", "RL Insights"])

# CHAT TAB ------------------------------------------------------------------
with chat_tab:
    st.subheader("AI Investing Copilot")
    st.caption("Ask anything — e.g., 'is AAPL a buy today' or 'allocate 10000 USD for me'.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Type your message…")

    # --- CSS fix to keep input pinned at bottom ---
    st.markdown(
        """
        <style>
        [data-testid="stChatInput"] {
            position: fixed;
            bottom: 1rem;
            width: 85%;
            left: 7%;
            z-index: 999;
            background-color: #111;
        }
        [data-testid="stChatMessageInputContainer"] {
            background: #111 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        # Detect tickers carefully (ignore normal words)
        tickers = [t for t in re.findall(r"\b[A-Z]{1,5}\b", user_msg.upper()) 
                   if t not in ["HELLO", "THANKS", "HEY", "HI", "BYE", "OK"]]

        context_lines = [
            f"Agent: {agent_choice}, Risk: {risk_tolerance}, Horizon: {horizon_years}y, Capital: {init_capital}, Income: {need_income}"
        ]

        px_df = None
        if tickers and YF_OK:
            try:
                px_df = yf.download(tickers, period="6mo", auto_adjust=True, progress=False)["Close"]
                changes = px_df.pct_change(5).iloc[-1].sort_values(ascending=False)
                rsis = {}
                for t in px_df.columns:
                    ret = px_df[t].pct_change()
                    gain = ret.where(ret > 0, 0.0)
                    loss = -ret.where(ret < 0, 0.0)
                    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
                    rsi = 100 - (100 / (1 + rs))
                    rsis[t] = round(float(rsi.iloc[-1]), 2)
                context_lines.append("5-day returns:\n" + changes.to_string())
                context_lines.append("RSI(14):\n" + pd.Series(rsis).to_string())
            except Exception as e:
                context_lines.append(f"(Failed to fetch live data: {e})")

        # Local data snapshot
        if st.session_state.prices_df is not None:
            last = st.session_state.prices_df.ffill().iloc[-1].dropna()
            context_lines.append("Local price snapshot:\n" + last.head(5).to_string())

        context = "\n\n".join(context_lines)

        system_prompt = textwrap.dedent("""
            You are a portfolio strategist and AI investment advisor.
            Use numeric data (returns, RSI) to identify momentum or overbought/oversold signals.
            Rules:
              - RSI < 30 → oversold → bullish
              - RSI > 70 → overbought → bearish
            Give short, specific insights with reasoning, not generic advice.
            Suggest tickers to buy/sell/hold with approximate weights that fit user's risk level.
        """)

        assistant_text = None
        try:
            if _openai_ready:
                with st.spinner("Analyzing markets..."):
                    if _openai_mode == "client_v1":
                        resp = _openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"User: {user_msg}\n\nContext:\n{context}"},
                            ],
                            temperature=0.2,
                            timeout=15,
                        )
                        assistant_text = resp.choices[0].message.content.strip()
                    else:
                        resp = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"User: {user_msg}\n\nContext:\n{context}"},
                            ],
                            temperature=0.2,
                            request_timeout=15,
                        )
                        assistant_text = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            st.warning(f"LLM unavailable: {e}")
            assistant_text = None

        if not assistant_text:
            assistant_text = _human_reply(user_msg)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

        if px_df is not None:
            fig = go.Figure()
            for t in px_df.columns:
                fig.add_trace(go.Scatter(x=px_df.index, y=px_df[t], name=t))
            fig.update_layout(title="Recent Price Trend", height=350)
            st.plotly_chart(fig, use_container_width=True)

# DATA EXPLORER -------------------------------------------------------------
with data_tab:
    st.subheader("Data Explorer")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Local Data"):
            try:
                prices, returns, _ = load_and_prepare_data(data_dir)
                st.session_state.prices_df = prices
                st.session_state.returns_df = returns
                st.success(f"Loaded {prices.shape[1]} tickers.")
            except Exception as e:
                st.error(f"Local data load failed: {e}")

        prices = st.session_state.prices_df
        if prices is not None:
            picks = st.multiselect("Select tickers", list(prices.columns), default=list(prices.columns)[:4])
            if picks:
                fig = go.Figure()
                for t in picks:
                    fig.add_trace(go.Scatter(x=prices.index, y=prices[t], name=t))
                fig.update_layout(height=400, title="Local Price History")
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        if YF_OK:
            tickers_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,AMZN,SPY")
            if st.button("Fetch Internet Data"):
                tickers = [x.strip().upper() for x in tickers_input.split(",") if x]
                df = fetch_internet_prices(tickers)
                if df is not None:
                    st.session_state.prices_df = df
                    st.session_state.returns_df = df.pct_change().dropna()
                    st.success("Fetched market data.")
                    st.line_chart(df)

# RL INSIGHTS ---------------------------------------------------------------
with rl_tab:
    st.subheader("RL Insights")
    if not _agents_ok:
        st.warning("RL agents not found. Ensure ppo_agent.py and dqn_agent.py exist.")
    else:
        prices = st.session_state.prices_df
        returns = st.session_state.returns_df
        if prices is not None and returns is not None:
            if agent_choice == "PPO":
                agent = PPOAgent(state_dim=returns.shape[1]*20, action_dim=returns.shape[1])
                ckpt = ppo_ckpt
            else:
                agent = DQNAgent(state_dim=returns.shape[1]*20, action_dim=5)
                ckpt = dqn_ckpt

            if Path(ckpt).exists():
                try:
                    agent.load(ckpt)
                    st.success(f"Loaded checkpoint: {ckpt}")
                except Exception as e:
                    st.warning(f"Failed to load checkpoint: {e}")
            else:
                st.info("No checkpoint found — using random weights.")

            state = returns.tail(20).values.flatten()
            action = agent.select_action(state, training=False)
            if agent_choice == "DQN":
                action = np.ones(returns.shape[1]) / returns.shape[1]
            else:
                action = np.maximum(action, 0)
                action = action / action.sum()

            out = pd.DataFrame({"Ticker": returns.columns, "Weight %": (action * 100).round(2)})
            st.dataframe(out)

            bt = run_backtest(prices, pd.Series(action, index=returns.columns))
            if not bt.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bt.index, y=bt["Cumulative"], name="Portfolio"))
                fig.update_layout(height=350, title="Backtest Performance")
                st.plotly_chart(fig, use_container_width=True)
