"""Standard trading-performance and risk metrics with explicit methodology."""

import numpy as np
import pandas as pd
TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    idx = pd.to_datetime(equity.index)
    years = (idx[-1] - idx[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS
    vol = returns.std(ddof=1)
    if vol == 0 or pd.isna(vol):
        return 0.0
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Sortino uses downside deviation relative to the daily risk-free target."""
    target = risk_free_rate / TRADING_DAYS
    excess = returns - target
    downside = np.minimum(excess, 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside)))
    if downside_deviation == 0 or pd.isna(downside_deviation):
        return 0.0
    return (excess.mean() / downside_deviation) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity: pd.Series):
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    trough_idx = drawdown.idxmin()
    return drawdown.min(), trough_idx, drawdown


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0,
                "gross_profit": 0.0, "gross_loss": 0.0}
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
        "avg_win": wins["pnl"].mean() if len(wins) else 0.0,
        "avg_loss": losses["pnl"].mean() if len(losses) else 0.0,
        "expectancy": trades["pnl"].mean(),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def full_report(equity: pd.Series, trades: pd.DataFrame, risk_free_rate: float = 0.065) -> tuple:
    daily_returns = equity.pct_change().dropna()
    max_dd, trough_idx, dd_series = max_drawdown(equity)
    report = {
        "starting_equity": float(equity.iloc[0]),
        "ending_equity": float(equity.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100),
        "cagr_pct": float(cagr(equity) * 100),
        "annual_volatility_pct": float(annualized_volatility(daily_returns) * 100),
        "sharpe_ratio": float(sharpe_ratio(daily_returns, risk_free_rate)),
        "sortino_ratio": float(sortino_ratio(daily_returns, risk_free_rate)),
        "max_drawdown_pct": float(max_dd * 100),
        "max_drawdown_date": str(trough_idx),
    }
    report.update(trade_stats(trades))
    return report, dd_series


def bootstrap_confidence_intervals(daily_returns: pd.Series, trades: pd.DataFrame,
                                    n_boot: int = 2000, ci: float = 0.90, seed: int = 42) -> dict:
    """IID bootstrap diagnostic only; it does not model time dependence."""
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2
    returns_arr = daily_returns.dropna().values
    if len(returns_arr) > 5:
        vals = np.empty(n_boot)
        for i in range(n_boot):
            vals[i] = sharpe_ratio(pd.Series(rng.choice(returns_arr, size=len(returns_arr), replace=True)))
        sharpe_lo, sharpe_hi = np.quantile(vals, [alpha, 1 - alpha])
    else:
        sharpe_lo = sharpe_hi = np.nan
    if not trades.empty and "pnl" in trades.columns:
        pnl = trades["pnl"].to_numpy()
        n = len(pnl)
        vals = np.array([rng.choice(pnl, size=n, replace=True).sum() for _ in range(n_boot)])
        pnl_lo, pnl_hi = np.quantile(vals, [alpha, 1 - alpha])
    else:
        pnl_lo = pnl_hi = np.nan
    return {"n_bootstrap_samples": n_boot, "confidence_level": ci,
            "sharpe_ci_low": float(sharpe_lo), "sharpe_ci_high": float(sharpe_hi),
            "total_pnl_ci_low_inr": float(pnl_lo), "total_pnl_ci_high_inr": float(pnl_hi),
            "n_trades_used": int(len(trades)) if not trades.empty else 0,
            "method": "IID bootstrap diagnostic; block bootstrap would better respect serial dependence"}
