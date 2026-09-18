"""
backtester.py  (audited portfolio backtester)
----------------------------------
Event-driven, bar-by-bar PORTFOLIO backtester. Three corrections vs. the
first submission, made after a self-review:

  FIX 3 (look-ahead bias): the original code detected a signal using day
  T's close and then opened the position AT day T's close -- a real bot
  cannot trade on information and price from the same instant. This
  version queues a signal detected on day T for EXECUTION AT DAY T+1'S
  OPEN, which is what a real next-session order would achieve. Stop-loss /
  take-profit orders remain intraday (they are resting orders that
  legitimately trigger within the same session once placed) but signal-
  based exits (trend reversal) are also deferred to the next open for the
  same reason as entries.

  FIX 4 (arbitrary ticker-priority ordering): when more entry signals fire
  on a day than there are free position slots, the original code filled
  slots in whatever order `self.tickers` happened to be listed -- an
  accident of the source code, not a real decision rule. This version
  ranks same-day candidates by signal strength (MACD histogram magnitude)
  and fills the strongest signals first.

  Cash accounting is explicit: only uninvested cash receives the modelled
  risk-free accrual, while invested capital receives stock-market returns.
"""

import pandas as pd
import numpy as np
import os

from strategy import generate_signals
from risk_manager import RiskManager


class Position:
    def __init__(self, ticker, entry_date, entry_price, qty, stop_loss, take_profit):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.qty = qty
        self.stop_loss = stop_loss
        self.take_profit = take_profit


class PortfolioBacktester:
    def __init__(self, data_dir, tickers, risk_manager: RiskManager, strategy_params=None,
                 preloaded_data=None, date_range=None, raw_data=None):
        self.data_dir = data_dir
        self.tickers = tickers
        self.rm = risk_manager
        self.strategy_params = strategy_params or {}
        self.date_range = date_range  # optional (start, end) to restrict TRADING/equity recording
        self.data = preloaded_data if preloaded_data is not None else {}
        self.raw_data = raw_data  # optional {ticker: raw OHLCV df}, avoids re-reading the workbook
        self.equity_curve = []
        self.trade_log = []
        self.open_positions = {}
        self.pending_entries = {}   # ticker -> {"atr_at_signal": float}
        self.pending_exits = set()  # tickers queued for next-open signal exit
        # Portfolio equity as of the last close, used to size NEW trades at
        # tomorrow's open (a real trader sizes off last night's marked
        # equity, not an intraday-updating figure) -- starts at starting cash.
        self.last_equity = self.rm.cash

    def load_data(self):
        if self.data:
            return
        if self.raw_data is None:
            # Fallback: load the consolidated workbook directly (single file,
            # one sheet per ticker -- see data_io.py). Only hit when a caller
            # constructs a PortfolioBacktester without pre-loading raw data;
            # callers running many backtests (grid searches) should load once
            # via data_io.load_raw_data_from_workbook() and pass raw_data=...
            # to avoid re-reading the workbook on every construction.
            from data_io import load_raw_data_from_workbook
            self.raw_data = load_raw_data_from_workbook(self.tickers)
        for t in self.tickers:
            df = generate_signals(self.raw_data[t].copy(), **self.strategy_params)
            df = df.set_index("date")
            self.data[t] = df

    def run(self):
        self.load_data()
        all_dates = sorted(set.union(*[set(df.index) for df in self.data.values()]))
        if self.date_range is not None:
            start, end = self.date_range
            trading_dates = [d for d in all_dates if start <= d <= end]
        else:
            trading_dates = all_dates

        for date in trading_dates:
            # 1) Execute entries queued from the PREVIOUS day's close-based signal,
            #    filled at TODAY'S open (no look-ahead)
            for ticker in list(self.pending_entries.keys()):
                df = self.data[ticker]
                if date not in df.index:
                    continue
                entry_price = df.loc[date, "open"]
                atr_value = self.pending_entries[ticker]["atr_at_signal"]
                qty, stop_loss, take_profit = self.rm.position_size(
                    entry_price, atr_value, portfolio_equity=self.last_equity
                )
                del self.pending_entries[ticker]
                if qty <= 0:
                    continue
                self.rm.open_position_cash_outflow(qty, entry_price)
                self.open_positions[ticker] = Position(ticker, date, entry_price, qty, stop_loss, take_profit)

            # 2) Execute signal-exits queued from the previous day, at TODAY'S open
            for ticker in list(self.pending_exits):
                self.pending_exits.discard(ticker)
                if ticker not in self.open_positions:
                    continue
                df = self.data[ticker]
                if date not in df.index:
                    continue
                exit_price = df.loc[date, "open"]
                self._close_position(ticker, date, exit_price, "signal_exit")

            # 3) Intraday stop-loss / take-profit checks (resting orders — legitimately
            #    same-day) for positions still open after step 2.
            #    ASSUMPTION (disclosed, not fixed): daily OHLC data cannot tell us
            #    WHICH of stop-loss or take-profit was touched first if both fall
            #    within a single day's high-low range (e.g. a gap day). This code
            #    always resolves stop-loss first, which is the conservative
            #    (return-understating, not overstating) assumption -- but it is a
            #    modeling choice, not a neutral fact, and is stated here explicitly.
            for ticker in list(self.open_positions.keys()):
                df = self.data[ticker]
                if date not in df.index:
                    continue
                row = df.loc[date]
                pos = self.open_positions[ticker]
                if row["low"] <= pos.stop_loss:
                    self._close_position(ticker, date, pos.stop_loss, "stop_loss")
                elif row["high"] >= pos.take_profit:
                    self._close_position(ticker, date, pos.take_profit, "take_profit")

            # 4) Evaluate TODAY'S close-based signals -> queue for execution tomorrow
            candidates = []
            for ticker in self.tickers:
                df = self.data[ticker]
                if date not in df.index:
                    continue
                row = df.loc[date]
                if ticker in self.open_positions:
                    if bool(row.get("long_exit_signal", False)):
                        self.pending_exits.add(ticker)
                else:
                    if ticker in self.pending_entries:
                        continue
                    if bool(row.get("long_entry", False)) and not pd.isna(row["atr14"]):
                        candidates.append((ticker, float(row.get("signal_strength", 0.0)), row["atr14"]))

            # Rank by signal strength (MACD histogram magnitude), strongest first,
            # and only queue as many as there is room for
            available_slots = self.rm.max_concurrent_positions - (len(self.open_positions) + len(self.pending_entries))
            candidates.sort(key=lambda x: x[1], reverse=True)
            for ticker, strength, atr_value in candidates[:max(available_slots, 0)]:
                self.pending_entries[ticker] = {"atr_at_signal": atr_value}

            # 5) Record end-of-day equity BEFORE overnight cash interest. This keeps
            # the first recorded observation equal to the configured starting capital.
            self._record_equity(date)

            # Cash earns the risk-free rate overnight. Do not accrue after the final
            # trading date because there is no next-day holding period in the test.
            if date != trading_dates[-1]:
                self.rm.accrue_daily_interest()
                # The next session's position sizing uses the prior close plus the
                # overnight cash accrual, so keep last_equity synchronized.
                self.last_equity = self._current_equity(date)

        # Close any positions still open at the end of the backtest, at final close.
        # Then replace the final equity observation with the post-liquidation value
        # so the ending equity, cash ledger and trade blotter reconcile exactly.
        last_date = trading_dates[-1]
        for ticker in list(self.open_positions.keys()):
            df = self.data[ticker]
            if last_date in df.index:
                self._close_position(ticker, last_date, df.loc[last_date, "close"], "end_of_backtest")

        if self.equity_curve:
            final_equity = self.rm.cash
            self.equity_curve[-1]["cash"] = self.rm.cash
            self.equity_curve[-1]["mark_to_market_equity"] = final_equity
            self.equity_curve[-1]["open_positions"] = 0
            self.last_equity = final_equity

        equity_df = pd.DataFrame(self.equity_curve).set_index("date")
        trades_df = pd.DataFrame(self.trade_log)
        return equity_df, trades_df

    def _close_position(self, ticker, date, exit_price, reason):
        pos = self.open_positions.pop(ticker)
        gross_pnl = (exit_price - pos.entry_price) * pos.qty
        exit_cost = self.rm.close_position_cash_inflow(pos.qty, exit_price)
        # Entry cost was already deducted from the cash ledger when the position
        # was opened. Do not subtract it a second time from trade-level P&L.
        entry_cost = self.rm.transaction_cost(pos.qty * pos.entry_price, "buy")
        net_pnl = gross_pnl - entry_cost - exit_cost
        holding_days = (date - pos.entry_date).days
        self.trade_log.append({
            "ticker": ticker,
            "entry_date": pos.entry_date,
            "exit_date": date,
            "entry_price": round(pos.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "qty": pos.qty,
            "holding_days": holding_days,
            "exit_reason": reason,
            "entry_cost": round(entry_cost, 2),
            "exit_cost": round(exit_cost, 2),
            "gross_pnl": round(gross_pnl, 2),
            "pnl": round(net_pnl, 2),
            "pnl_pct": round((exit_price / pos.entry_price - 1) * 100, 2),
        })

    def _current_equity(self, date):
        positions_market_value = 0.0
        for ticker, pos in self.open_positions.items():
            df = self.data[ticker]
            if date in df.index:
                positions_market_value += df.loc[date, "close"] * pos.qty
        return self.rm.cash + positions_market_value

    def _record_equity(self, date):
        # Cash is already net of every open position's notional (deducted at
        # entry), so mark-to-market equity is cash + the CURRENT market
        # value of open positions -- not cash + unrealized P&L on top of an
        # equity figure that still implicitly contains the entry notional.
        mtm_equity = self._current_equity(date)
        self.equity_curve.append({
            "date": date,
            "cash": self.rm.cash,
            "mark_to_market_equity": mtm_equity,
            "open_positions": len(self.open_positions),
        })
        self.last_equity = mtm_equity  # used to size tomorrow's new entries


def run_backtest(data_dir, tickers, out_dir, strategy_params=None, risk_kwargs=None):
    rm = RiskManager(**(risk_kwargs or {}))
    bt = PortfolioBacktester(data_dir, tickers, rm, strategy_params=strategy_params)
    equity_df, trades_df = bt.run()

    os.makedirs(out_dir, exist_ok=True)
    equity_df.to_csv(os.path.join(out_dir, "equity_curve.csv"))
    trades_df.to_csv(os.path.join(out_dir, "trade_log.csv"), index=False)

    return equity_df, trades_df, rm
