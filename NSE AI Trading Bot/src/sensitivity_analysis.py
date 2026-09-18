"""
sensitivity_analysis.py  (v3 — in-sample only)
--------------------------------------------------
This grid is a robustness check on PARAMETER CHOICE, not a second
performance claim -- so it must run only on the same in-sample window
walk_forward.py uses for parameter selection. Earlier versions ran this
grid over the FULL data window, which silently included the exact dates
walk_forward.py later called "out-of-sample, never touched during
selection." That overlap meant the two sections weren't actually
independent, even though the report implied they were. Restricting this
grid to IN_SAMPLE_END (imported from universe.py, the single shared
source of truth for the split) fixes that.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from backtester import PortfolioBacktester
from risk_manager import RiskManager
from metrics import full_report
from universe import TICKERS, IN_SAMPLE_END

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
LOGS_DIR = os.path.join(BASE, "logs")

RISK_GRID = [(1.5, 3.0), (2.0, 4.0), (2.5, 5.0), (2.0, 2.0), (3.0, 3.0)]
ENTRY_GRID = [
    (20, 50, 40, 70),   # original / baseline
    (10, 30, 40, 70),   # faster trend detection
    (20, 50, 30, 60),   # looser RSI band
    (15, 40, 45, 75),   # mid-speed, shifted RSI band
]


def load_all_data():
    from data_io import load_raw_data_from_workbook
    return load_raw_data_from_workbook(TICKERS)  # ONE workbook read, reused for all 20 grid combos below


def main():
    raw_data = load_all_data()
    all_dates = sorted(set.union(*[set(df["date"]) for df in raw_data.values()]))
    in_sample_range = (all_dates[0], IN_SAMPLE_END)
    print(f"Running sensitivity grid on IN-SAMPLE window only: "
          f"{in_sample_range[0].date()} -> {in_sample_range[1].date()}")
    print("(This intentionally excludes the out-of-sample period walk_forward.py evaluates.)\n")

    rows = []
    for ema_fast, ema_slow, rsi_lo, rsi_hi in ENTRY_GRID:
        strategy_params = {"ema_fast": ema_fast, "ema_slow": ema_slow,
                            "rsi_entry_low": rsi_lo, "rsi_entry_high": rsi_hi}
        for sl_mult, tp_mult in RISK_GRID:
            rm = RiskManager(stop_loss_atr_mult=sl_mult, take_profit_atr_mult=tp_mult)
            bt = PortfolioBacktester(DATA_DIR, TICKERS, rm,
                                      strategy_params=strategy_params,
                                      preloaded_data=None, date_range=in_sample_range,
                                      raw_data=raw_data)
            # raw_data (loaded ONCE above from the single workbook) is reused across
            # all 20 grid combinations; load_data() inside run() applies
            # generate_signals(raw_data[t], **strategy_params) per combo without
            # re-reading the workbook from disk each time.
            equity_df, trades_df = bt.run()
            if equity_df.empty:
                continue
            report, _ = full_report(equity_df["mark_to_market_equity"], trades_df)
            rows.append({
                "ema_fast": ema_fast, "ema_slow": ema_slow,
                "rsi_low": rsi_lo, "rsi_high": rsi_hi,
                "stop_atr_mult": sl_mult, "target_atr_mult": tp_mult,
                "total_return_pct": round(report["total_return_pct"], 2),
                "sharpe_ratio": round(report["sharpe_ratio"], 2),
                "max_drawdown_pct": round(report["max_drawdown_pct"], 2),
                "total_trades": report["total_trades"],
                "win_rate_pct": round(report["win_rate"] * 100, 1),
                "profit_factor": round(report["profit_factor"], 2) if report["profit_factor"] != float("inf") else "inf",
            })
            print(f"ema({ema_fast}/{ema_slow}) rsi[{rsi_lo}-{rsi_hi}] "
                  f"stop/target({sl_mult}/{tp_mult}): "
                  f"return={rows[-1]['total_return_pct']}% sharpe={rows[-1]['sharpe_ratio']}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(LOGS_DIR, "sensitivity_analysis.csv")
    df.to_csv(out_path, index=False)

    n_positive = (df["total_return_pct"] > 0).sum()
    print(f"\n{len(df)} parameter combinations tested (IN-SAMPLE ONLY). "
          f"{n_positive} produced a positive total return ({n_positive/len(df)*100:.0f}%).")
    print(f"Best combination by Sharpe: \n{df.loc[df['sharpe_ratio'].idxmax()]}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
