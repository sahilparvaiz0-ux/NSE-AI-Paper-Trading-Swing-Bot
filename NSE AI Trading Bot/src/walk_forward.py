"""
walk_forward.py
-----------------
The first submission ran one continuous in-sample backtest with hand-picked
parameters and called the sensitivity grid a "robustness check" — but
picking parameters and evaluating them on the SAME data is not genuine
out-of-sample validation, even if the grid is wide. This script fixes that:

  1. Split the ~2-year window into an IN-SAMPLE period (first ~15 months,
     used only to select parameters) and an OUT-OF-SAMPLE period (the
     remaining ~9 months, used only to report results).
  2. Run the full (entry x risk) parameter grid on the in-sample period only,
     pick the combination with the best in-sample Sharpe ratio.
  3. Re-run ONCE on the out-of-sample period using those chosen parameters
     (fresh starting capital, but with indicators warmed up on the full
     price history so EMA/RSI/MACD aren't distorted by a cold start) and
     report that as the genuine held-out result.

This is a standard train/test discipline, not a full walk-forward
re-optimization loop (which would re-fit periodically), but it removes the
single biggest validity gap in the original submission: parameters chosen
and graded on the same data.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from strategy import generate_signals
from backtester import PortfolioBacktester
from risk_manager import RiskManager
from metrics import full_report
from universe import TICKERS, IN_SAMPLE_END, OUT_OF_SAMPLE_START
from sensitivity_analysis import RISK_GRID, ENTRY_GRID

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
LOGS_DIR = os.path.join(BASE, "logs")





def load_all_data():
    from data_io import load_raw_data_from_workbook
    return load_raw_data_from_workbook(TICKERS)  # ONE workbook read, reused for all 20+1 backtests below


def run_with_params(raw_data, strategy_params, risk_kwargs, date_range):
    # generate_signals uses the FULL price history (so indicators are
    # properly warmed up) but the backtester only trades/records equity
    # within date_range -- this avoids a cold-start distortion at the
    # start of the out-of-sample window.
    prepared = {}
    for t, df in raw_data.items():
        sig_df = generate_signals(df.copy(), **strategy_params).set_index("date")
        prepared[t] = sig_df
    rm = RiskManager(**risk_kwargs)
    bt = PortfolioBacktester(DATA_DIR, TICKERS, rm, preloaded_data=prepared, date_range=date_range)
    equity_df, trades_df = bt.run()
    return equity_df, trades_df, rm


def main():
    raw_data = load_all_data()
    all_dates = sorted(set.union(*[set(df["date"]) for df in raw_data.values()]))
    in_sample_range = (all_dates[0], IN_SAMPLE_END)
    out_sample_range = (OUT_OF_SAMPLE_START, all_dates[-1])

    print(f"In-sample:      {in_sample_range[0].date()} -> {in_sample_range[1].date()}")
    print(f"Out-of-sample:  {out_sample_range[0].date()} -> {out_sample_range[1].date()}")

    # 1) Grid search on IN-SAMPLE data only
    results = []
    for ema_fast, ema_slow, rsi_lo, rsi_hi in ENTRY_GRID:
        strategy_params = {"ema_fast": ema_fast, "ema_slow": ema_slow,
                            "rsi_entry_low": rsi_lo, "rsi_entry_high": rsi_hi}
        for sl_mult, tp_mult in RISK_GRID:
            risk_kwargs = {"stop_loss_atr_mult": sl_mult, "take_profit_atr_mult": tp_mult}
            equity_df, trades_df, rm = run_with_params(raw_data, strategy_params, risk_kwargs, in_sample_range)
            if equity_df.empty:
                continue
            report, _ = full_report(equity_df["mark_to_market_equity"], trades_df)
            results.append({
                "strategy_params": strategy_params, "risk_kwargs": risk_kwargs,
                "in_sample_sharpe": report["sharpe_ratio"],
                "in_sample_return_pct": report["total_return_pct"],
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(LOGS_DIR, "walk_forward_in_sample_grid.csv"), index=False)
    best = results_df.loc[results_df["in_sample_sharpe"].idxmax()]
    print(f"\nBest IN-SAMPLE parameters (by Sharpe): {best['strategy_params']} / {best['risk_kwargs']}")
    print(f"In-sample Sharpe: {best['in_sample_sharpe']:.2f}, return: {best['in_sample_return_pct']:.2f}%")

    # 2) Evaluate chosen parameters ONCE on the held-out OUT-OF-SAMPLE period
    oos_equity, oos_trades, oos_rm = run_with_params(
        raw_data, best["strategy_params"], best["risk_kwargs"], out_sample_range)
    oos_report, _ = full_report(oos_equity["mark_to_market_equity"], oos_trades)

    print("\n=== OUT-OF-SAMPLE RESULT (held-out, never used for parameter selection) ===")
    for k in ["total_return_pct", "cagr_pct", "sharpe_ratio", "sortino_ratio",
              "max_drawdown_pct", "total_trades", "win_rate", "profit_factor"]:
        print(f"{k:24s}: {oos_report[k]}")

    oos_summary = {
        "chosen_strategy_params": best["strategy_params"],
        "chosen_risk_params": best["risk_kwargs"],
        "in_sample_sharpe": best["in_sample_sharpe"],
        "in_sample_return_pct": best["in_sample_return_pct"],
        **{f"oos_{k}": v for k, v in oos_report.items()},
    }
    pd.Series(oos_summary).to_csv(os.path.join(LOGS_DIR, "walk_forward_oos_result.csv"))
    oos_trades.to_csv(os.path.join(LOGS_DIR, "walk_forward_oos_trades.csv"), index=False)
    print(f"\nSaved -> {LOGS_DIR}/walk_forward_oos_result.csv")


if __name__ == "__main__":
    main()
