"""
strategy.py  (v2 — parameterized)
-----------------------------------
Swing-trading strategy: "EMA Trend + RSI/MACD Confirmation".

v2 change: all thresholds are now function arguments with the original
defaults, instead of hardcoded constants. This was needed because the first
submission's robustness check (sensitivity_analysis.py) only varied the
RISK parameters (stop/target ATR multiples) and never varied the ENTRY
LOGIC parameters (EMA lengths, RSI bands) -- so the claim "the
underperformance isn't a parameter-tuning artifact" was only proven for
half the parameter space. This version lets both be swept together.

Also adds `signal_strength`: a continuous score (MACD histogram, in price
units) used by the backtester to RANK same-day entry candidates when more
signals fire than there are free position slots, replacing the old
behaviour of filling slots in a fixed, arbitrary ticker-list order.
"""

import pandas as pd
from indicators import add_all_indicators


def generate_signals(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    rsi_entry_low: float = 40,
    rsi_entry_high: float = 70,
    rsi_overbought_exit: float = 80,
) -> pd.DataFrame:
    df = add_all_indicators(df, ema_fast=ema_fast, ema_slow=ema_slow)
    df["ema_cross_up"] = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    df["ema_cross_down"] = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))

    df["long_entry"] = (
        df["ema_cross_up"]
        & (df["rsi14"] >= rsi_entry_low) & (df["rsi14"] <= rsi_entry_high)
        & (df["macd_hist"] > 0)
    )
    df["long_exit_signal"] = df["ema_cross_down"] | (df["rsi14"] > rsi_overbought_exit)

    # Continuous strength score for ranking same-day entry candidates
    # (used only when the entry filter above is already True).
    #
    # FIX 15: macd_hist is in absolute price units, so an unnormalized score
    # systematically favours high-priced stocks (e.g. TCS ~Rs 4,000-4,500,
    # LT ~Rs 3,700-4,000) over low-priced ones (e.g. ITC ~Rs 400-430) for
    # equivalent RELATIVE momentum -- the "principled" ranking rule was
    # actually a proxy for share price, not signal strength. Normalizing by
    # price makes it a relative (percentage-of-price) momentum score, which
    # is comparable across tickers regardless of face value.
    df["signal_strength"] = (df["macd_hist"] / df["close"]).clip(lower=0)
    return df
