"""
main.py
-------
End-to-end pipeline:
  1. (Re)generate synthetic NSE data if not present
  2. Run the portfolio backtest for the swing-trading bot
  3. Compute performance/risk metrics
  4. Produce charts and a machine-readable summary for the report
"""

import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from backtester import run_backtest
from metrics import full_report, bootstrap_confidence_intervals
from universe import TICKERS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
CHARTS_DIR = os.path.join(BASE, "charts")
LOGS_DIR = os.path.join(BASE, "logs")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

TICKERS = TICKERS  # imported directly from universe.py


def plot_equity_curve(equity_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_df.index, equity_df["mark_to_market_equity"], color="#1a5276", linewidth=1.6, label="Mark-to-Market Equity")
    ax.axhline(equity_df["mark_to_market_equity"].iloc[0], color="gray", linestyle="--", linewidth=1, label="Starting Capital")
    ax.set_title("Swing-Trading Bot — Portfolio Equity Curve (Nifty-50 Basket, Real NSE Data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (INR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdown(dd_series, out_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(dd_series.index, dd_series.values * 100, 0, color="#c0392b", alpha=0.5)
    ax.set_title("Portfolio Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trade_pnl_distribution(trades_df, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#27ae60" if p > 0 else "#c0392b" for p in trades_df["pnl"]]
    ax.bar(range(len(trades_df)), trades_df["pnl"], color=colors)
    ax.set_title("Per-Trade Net P&L")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Net P&L (INR)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_signals(ticker, data_dir, out_path):
    from strategy import generate_signals
    from data_io import load_raw_data_from_workbook
    df = load_raw_data_from_workbook([ticker])[ticker]
    df = generate_signals(df).set_index("date")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["close"], color="#333333", linewidth=1, label="Close")
    ax.plot(df.index, df["ema20"], color="#2980b9", linewidth=1, label="EMA20")
    ax.plot(df.index, df["ema50"], color="#e67e22", linewidth=1, label="EMA50")
    entries = df[df["long_entry"]]
    ax.scatter(entries.index, entries["close"], marker="^", color="#27ae60", s=70, zorder=5, label="Long Entry Signal")
    exits = df[df["long_exit_signal"]]
    ax.scatter(exits.index, exits["close"], marker="v", color="#c0392b", s=50, zorder=5, alpha=0.6, label="Exit Signal")
    ax.set_title(f"{ticker} — EMA Trend + RSI/MACD Confirmation Strategy (Signals)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_win_loss_by_ticker(trades_df, out_path):
    summary = trades_df.groupby("ticker")["pnl"].agg(["sum", "count"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#27ae60" if v > 0 else "#c0392b" for v in summary["sum"]]
    ax.bar(summary.index, summary["sum"], color=colors)
    ax.set_title("Net P&L by Stock")
    ax.set_ylabel("Net P&L (INR)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(DATA_DIR, "nifty_data.xlsx")):
        print("No consolidated workbook found. Run real_data_loader.py first (or data_generator.py for the legacy synthetic fallback).")
        from real_data_loader import main as load_main
        load_main()

    print("Running portfolio backtest...")
    equity_df, trades_df, rm = run_backtest(DATA_DIR, TICKERS, LOGS_DIR)
    equity_series = equity_df["mark_to_market_equity"]

    report, dd_series = full_report(equity_series, trades_df)
    report["universe"] = TICKERS
    report["strategy"] = "EMA20/EMA50 Trend + RSI(14) + MACD Histogram Confirmation (Swing, long-only)"
    report["starting_capital_inr"] = rm.starting_capital
    report["risk_per_trade_pct"] = rm.risk_per_trade_pct * 100
    report["max_concurrent_positions"] = rm.max_concurrent_positions
    report["cash_interest_earned_inr"] = rm.interest_earned
    calendar_years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
    cash_only_end = rm.starting_capital * (1 + rm.annual_risk_free_rate) ** calendar_years
    report["cash_baseline_annual_rate_pct"] = rm.annual_risk_free_rate * 100
    report["calendar_years"] = calendar_years
    report["cash_only_baseline_end_inr"] = cash_only_end
    report["cash_only_baseline_return_pct"] = (cash_only_end / rm.starting_capital - 1) * 100
    report["bot_increment_vs_cash_baseline_inr"] = equity_series.iloc[-1] - cash_only_end

    daily_returns = equity_series.pct_change().dropna()
    ci = bootstrap_confidence_intervals(daily_returns, trades_df, n_boot=2000, ci=0.90)
    report["bootstrap_ci"] = ci
    print(f"\n90% bootstrap CI -- Sharpe: [{ci['sharpe_ci_low']:.2f}, {ci['sharpe_ci_high']:.2f}], "
          f"Total P&L: [Rs {ci['total_pnl_ci_low_inr']:,.0f}, Rs {ci['total_pnl_ci_high_inr']:,.0f}] "
          f"(n={ci['n_trades_used']} trades)")

    with open(os.path.join(LOGS_DIR, "performance_summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== PERFORMANCE SUMMARY ===")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k:28s}: {v:,.2f}")
        else:
            print(f"{k:28s}: {v}")

    print("\nGenerating charts...")
    plot_equity_curve(equity_df, os.path.join(CHARTS_DIR, "01_equity_curve.png"))
    plot_drawdown(dd_series, os.path.join(CHARTS_DIR, "02_drawdown.png"))
    if not trades_df.empty:
        plot_trade_pnl_distribution(trades_df, os.path.join(CHARTS_DIR, "03_trade_pnl.png"))
        plot_win_loss_by_ticker(trades_df, os.path.join(CHARTS_DIR, "04_pnl_by_ticker.png"))
    plot_sample_signals("RELIANCE", DATA_DIR, os.path.join(CHARTS_DIR, "05_signals_RELIANCE.png"))
    plot_sample_signals("HDFCBANK", DATA_DIR, os.path.join(CHARTS_DIR, "06_signals_HDFCBANK.png"))

    trades_df.to_csv(os.path.join(LOGS_DIR, "trade_log.csv"), index=False)
    print(f"\nDone. Charts -> {CHARTS_DIR}\nLogs -> {LOGS_DIR}")


if __name__ == "__main__":
    main()
