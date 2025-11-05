# 05_GenAI_Explainability_Dashboard.py
# Smart Portfolio Allocator — RL + GenAI Dashboard
# Streamlit app with four tabs: Chat, Data Explorer, RL Insights, Stock Intelligence

from __future__ import annotations

import os
import sys
import json
import time
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Optional vendors (handled safely)
# -----------------------------
try:
    import yfinance as yf
except Exception:  # if yfinance not installed yet
    yf = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # LLM is optional; we provide an offline fallback

# Make project root importable so "src.agents" works when launched from /notebooks
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -----------------------------
# Attempt to import RL agents (safe)
# -----------------------------
PPOAgent = None
DQNAgent = None

def _try_import_agents() -> Tuple[Optional[type], Optional[type], Optional[str]]:
    """
    Try multiple import paths for PPO/DQN agents. Return (PPOAgent, DQNAgent, err_msg)
    without throwing import errors into the UI.
    """
    try_paths = [
        "src.agents.ppo_agent",
        "agents.ppo_agent",
    ]
    err = None
    ppo_cls = None
    dqn_cls = None
    for mod in try_paths:
        try:
            p = __import__(mod, fromlist=["PPOAgent"])
            ppo_cls = getattr(p, "PPOAgent", None)
            if ppo_cls:
                break
        except Exception as e:
            err = str(e)

    try_paths = [
        "src.agents.dqn_agent",
        "agents.dqn_agent",
    ]
    for mod in try_paths:
        try:
            d = __import__(mod, fromlist=["DQNAgent"])
            dqn_cls = getattr(d, "DQNAgent", None)
            if dqn_cls:
                break
        except Exception as e:
            err = str(e)

    return ppo_cls, dqn_cls, err


PPOAgent, DQNAgent, _AGENT_IMPORT_ERR = _try_import_agents()

# ============================================================================
# Utilities
# ============================================================================

@st.cache_data(show_spinner=False)
def _normalize_tickers(raw: str) -> List[str]:
    tks = [t.strip().upper() for t in raw.split(",") if t.strip()]
    # remove duplicates while preserving order
    seen = set()
    out = []
    for t in tks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _safe_col(series: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """
    Ensure we return a 1-D float series. If a DF comes in, take the first column.
    """
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            raise ValueError("Empty DataFrame passed for column selection.")
        series = series.iloc[:, 0]
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s.name = name
    return s


@st.cache_data(show_spinner=False)
def fetch_from_yf(ticker: str, period: str = "1y") -> Optional[pd.Series]:
    """
    Fetch adjusted close from Yahoo Finance with fallbacks.
    Returns a price Series indexed by DatetimeIndex, named as ticker.
    """
    if yf is None:
        return None
    try:
        # Prefer auto_adjust=True to get 'Close' already adjusted
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None

        # With auto_adjust=True, 'Close' is adjusted and is present for single ticker
        if "Close" in df.columns:
            s = _safe_col(df["Close"], ticker)
            return s.dropna()

        # If multi-index (unlikely for single ticker) – fallback
        if isinstance(df.columns, pd.MultiIndex):
            # Try ('Close', ticker) or (ticker, 'Close')
            try_keys = [( "Close", ticker), (ticker, "Close") ]
            for key in try_keys:
                if key in df.columns:
                    return _safe_col(df[key], ticker).dropna()
        return None
    except Exception:
        return None


def _read_local_csv_one(data_dir: Path, ticker: str) -> Optional[pd.Series]:
    """
    Try reading a local CSV named like '{ticker}.csv' or any CSV that has a 'Close'/'Adj Close' column.
    """
    data_dir = Path(data_dir)
    candidates = []
    # Exact-name first if present
    exact = data_dir / f"{ticker}.csv"
    if exact.exists():
        candidates.append(exact)
    # General fallback: any csv containing ticker in name
    candidates.extend(sorted(data_dir.glob(f"*{ticker}*.csv")))

    for fp in candidates:
        try:
            df = pd.read_csv(fp)
            # Try common date columns
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.set_index("Date").sort_index()
            elif df.columns[0].lower() in ("date", "time", "timestamp"):
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
                df = df.set_index(df.columns[0]).sort_index()
            else:
                # try to parse first column as date
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
                df = df.set_index(df.columns[0]).sort_index()

            for c in ["Adj Close", "Close", "close", "adj_close", "Price", "price"]:
                if c in df.columns:
                    return _safe_col(df[c], ticker).dropna()
            # uncommon: price might be the only numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 1:
                return _safe_col(df[numeric_cols[0]], ticker).dropna()
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_prices_hybrid(tickers: List[str], data_dir: str, period: str = "1y") -> pd.DataFrame:
    """
    For each ticker, load from local CSV if available; otherwise fetch from Yahoo.
    Returns a simple DataFrame with columns = tickers (no MultiIndex),
    index as datetime, float values.
    """
    data_dir = Path(data_dir) if data_dir else Path(".")
    cols = {}
    missing = []

    for tk in tickers:
        s_local = None
        if data_dir.exists():
            s_local = _read_local_csv_one(data_dir, tk)
        if s_local is not None and not s_local.empty:
            cols[tk] = s_local.astype(float)
            continue

        s_web = fetch_from_yf(tk, period=period)
        if s_web is not None and not s_web.empty:
            cols[tk] = s_web.astype(float)
            continue

        missing.append(tk)

    if not cols:
        raise ValueError("No valid data loaded for given tickers.")

    df = pd.concat(cols.values(), axis=1)
    df.columns = list(cols.keys())
    df = df.sort_index().dropna(how="all")
    if df.empty:
        raise ValueError("Data loaded but empty after cleaning.")
    return df


def compute_basic_indicators(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute daily returns, 20/50 SMA, and a simple RSI(14) for each column.
    """
    rets = df.pct_change().dropna()

    sma20 = df.rolling(20).mean()
    sma50 = df.rolling(50).mean()

    # RSI
    delta = df.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / (roll_down + 1e-9)
    RSI = 100.0 - (100.0 / (1.0 + RS))

    return {
        "returns": rets,
        "sma20": sma20,
        "sma50": sma50,
        "rsi14": RSI,
    }


def naive_forecast_series(s: pd.Series, horizon: int = 30) -> Tuple[float, float]:
    """
    Very simple 'forecast': last price +/- 1 std of last 60 returns scaled by sqrt(horizon).
    Returns (low, high). If insufficient data, returns last price bounds of ±2%.
    """
    s = s.dropna()
    if len(s) < 10:
        p = float(s.iloc[-1])
        return p * 0.98, p * 1.02
    last_price = float(s.iloc[-1])
    rets = s.pct_change().dropna().tail(60)
    if rets.empty:
        return last_price * 0.98, last_price * 1.02
    vol = float(rets.std())
    band = last_price * vol * math.sqrt(max(horizon, 1))
    return last_price - band, last_price + band


def _fmt_money(x: float, cur: str = "$") -> str:
    try:
        return f"{cur}{x:,.2f}"
    except Exception:
        return str(x)


# ============================================================================
# UI — Sidebar
# ============================================================================
st.set_page_config(page_title="GenAI + RL Assistant", layout="wide")
st.sidebar.title("GenAI + RL Assistant")

rl_agent_choice = st.sidebar.selectbox("RL Agent", ["PPO", "DQN"], index=0)
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Internet (yfinance)"],  # placeholder for future sources
    index=0,
)
local_dir = st.sidebar.text_input(
    "Local data directory",
    value=str(Path.home() / "Downloads"),
    help="Used for CSV fallbacks; put your local .csv files here if desired.",
)

st.sidebar.markdown("**RL Checkpoints**")
ppo_ckpt = st.sidebar.text_input("ppo_agent.pth", value="ppo_agent.pth")
dqn_ckpt = st.sidebar.text_input("dqn_agent.pth", value="dqn_agent.pth")

# ============================================================================
# Tabs
# ============================================================================
tabs = st.tabs(["Chat", "Data Explorer", "RL Insights", "Stock Intelligence"])

# =============================================================================
# TAB 1 — AI Investing Copilot (Capstone-aligned)
# =============================================================================
with tabs[0]:
    st.header("AI Investing Copilot")
    st.markdown(
        """
        Ask portfolio-related questions such as:
        - "Explain why PPO chose AAPL over MSFT"
        - "Compare PPO vs DQN performance this week"
        - "How should I allocate $50,000 now?"
        """
    )

    user_query = st.text_input("Enter your message", key="chat_q")

    if st.button("Analyze", key="chat_btn"):
        if not user_query.strip():
            st.warning("Please enter a query first.")
        else:
            st.write(f"**User:** {user_query}")

            # Basic intent detection
            qlow = user_query.lower()
            wants_rl = any(w in qlow for w in ["ppo", "dqn", "agent", "policy", "q-network", "sharpe"])

            # Try to load agents and provide simple metrics if available
            rl_summary = {}
            try:
                if wants_rl and (PPOAgent or DQNAgent):
                    # Minimal "fake" setup for metric illustration.
                    # In your capstone, replace these with real evaluation metrics.
                    rl_summary = {
                        "ppo": {"avg_reward_30d": 0.054, "sharpe": 1.22},
                        "dqn": {"avg_reward_30d": 0.041, "sharpe": 1.08},
                        "note": "Replace with real logs/metrics from your training runs.",
                    }
                    st.json(rl_summary)
                elif wants_rl and not (PPOAgent or DQNAgent):
                    st.info("PPO/DQN Python modules not importable yet. RL insights limited to reasoning layer.")
            except Exception as e:
                st.info(f"RL metrics unavailable: {e}")

            # LLM reasoning layer
            response_text = ""
            if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    prompt = f"""
                    You are FinGPT — a reasoning engine inside a reinforcement learning portfolio dashboard.
                    The user asked: {user_query}
                    RL analysis output (if any): {json.dumps(rl_summary)}
                    Provide a clear, concise explanation grounded in risk-reward logic and portfolio construction.
                    """
                    resp = client.chat.completions.create(
                        model="gpt-5",
                        messages=[
                            {"role": "system", "content": "You are FinGPT, a financial analyst AI that interprets RL-driven portfolio allocations."},
                            {"role": "user", "content": textwrap.dedent(prompt)},
                        ],
                        temperature=0.4,
                        max_tokens=450,
                    )
                    response_text = resp.choices[0].message.content.strip()
                except Exception as e:
                    response_text = f"LLM disabled or failed: {e}. Here is a structured, non-LLM answer:\n"
            else:
                response_text = "LLM is not configured. Presenting a rule-based explanation.\n"

            if not response_text:
                response_text = "No reasoning generated."

            # Simple rule-based fallback if LLM not available
            if "LLM is not configured" in response_text or response_text.startswith("LLM disabled"):
                if wants_rl:
                    response_text += (
                        "\n• PPO typically outputs continuous weights and tends to be smoother in allocation updates.\n"
                        "• DQN uses discrete actions (e.g., strategy buckets) and may react more abruptly.\n"
                        "• If average rewards and Sharpe are higher for PPO in your recent backtests, prefer PPO allocations.\n"
                        "• In volatile periods, reduce weights to high-beta assets and rebalance more frequently."
                    )
                else:
                    response_text += (
                        "\n• For general market questions, combine recent price trend, realized volatility, and macro signals.\n"
                        "• Use a simple momentum + mean-reversion blend to form a baseline view while you evaluate RL outputs."
                    )

            st.markdown(f"**FinGPT:**\n\n{response_text}")

# =============================================================================
# TAB 2 — Data Explorer
# =============================================================================
with tabs[1]:
    st.header("Data Explorer")

    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_text = st.text_input("Ticker(s), comma separated", value="AAPL, MSFT", key="de_tks")
    with col2:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1, key="de_period")

    if st.button("Load Local CSVs / Internet", key="de_btn"):
        tks = _normalize_tickers(tickers_text)
        try:
            df = load_prices_hybrid(tks, local_dir, period=period)
            st.success("Loaded price data")
            st.dataframe(df.tail(10))
            st.line_chart(df)
            st.caption("If local CSVs are present they are preferred; otherwise Yahoo is used.")
        except Exception as e:
            st.error(f"Failed to load data: {e}")

# =============================================================================
# TAB 3 — RL Insights
# =============================================================================
with tabs[2]:
    st.header("RL Insights")

    if not (PPOAgent or DQNAgent):
        st.warning(
            "ppo_agent.py / dqn_agent.py missing or failed to import.\n"
            f"Import error hint: {_AGENT_IMPORT_ERR or 'N/A'}"
        )
    else:
        st.success("RL agent modules importable.")
    st.caption("For a full demo, point the checkpoints below to real trained models.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("PPO")
        ppo_path = Path(local_dir) / ppo_ckpt
        if ppo_path.exists():
            st.write(f"Found: {ppo_path}")
            st.caption("Loading is stubbed; replace with real load/eval code in your capstone.")
        else:
            st.info(f"{ppo_path} not found.")
    with c2:
        st.subheader("DQN")
        dqn_path = Path(local_dir) / dqn_ckpt
        if dqn_path.exists():
            st.write(f"Found: {dqn_path}")
            st.caption("Loading is stubbed; replace with real load/eval code in your capstone.")
        else:
            st.info(f"{dqn_path} not found.")

    st.markdown("---")
    st.markdown("**Next steps for capstone integration**")
    st.markdown(
        "- Load checkpoints into PPOAgent/DQNAgent and compute rolling Sharpe, drawdown, and reward curves.\n"
        "- Surface the agent’s current recommended weights for a selected universe.\n"
        "- Feed those outputs into the Chat tab for explanation."
    )

# =============================================================================
# TAB 4 — Stock Intelligence
# =============================================================================
with tabs[3]:
    st.header("Stock Intelligence")
    st.caption("Analyze stock trends, compute returns, and visualize performance with automatic data fetching and a naive forecast.")

    tks_text = st.text_input("Ticker(s), comma separated", value="AAPL, MSFT", key="si_tks")
    horizon = st.slider("Forecast horizon (business days)", 5, 90, 30, key="si_hor")

    if st.button("Analyze Forecast", key="si_btn"):
        tks = _normalize_tickers(tks_text)
        errors = []
        try:
            df = load_prices_hybrid(tks, local_dir, period="1y")
        except Exception as e:
            st.error(f"Error: {e}")
            df = None

        if df is not None:
            st.success("Data fetched successfully.")
            st.line_chart(df)

            inds = compute_basic_indicators(df)

            # Show a neat metrics block per ticker
            for tk in df.columns:
                with st.expander(f"{tk} — summary"):
                    s = df[tk].dropna()
                    if len(s) < 2:
                        st.info("Not enough data to summarize.")
                        continue

                    cur_price = float(s.iloc[-1])
                    low, high = naive_forecast_series(s, horizon=horizon)
                    m1 = inds["returns"][tk].tail(21).mean() * 100.0
                    vol = inds["returns"][tk].tail(60).std() * math.sqrt(252) * 100.0  # annualized

                    st.markdown(
                        f"""
                        - Current price: **{_fmt_money(cur_price)}**  
                        - 1-month average daily return: **{m1:.2f}%**  
                        - Annualized volatility (60d): **{vol:.2f}%**  
                        - Expected price range in ~{horizon} business days: **{_fmt_money(low)} – {_fmt_money(high)}**
                        """
                    )

            st.markdown("---")
            st.caption("Forecast is a simple volatility-scaled band for illustration. Replace with your model of choice for the capstone.")

        # Show any collected errors
        for e in errors:
            st.warning(e)

# Footer
st.markdown("---")
st.caption("GenAI + RL Assistant — Capstone Dashboard. Local CSVs preferred; Yahoo used as a fallback. RL modules and checkpoints are optional to run the app.")
