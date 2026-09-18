"""
data_generator.py  (v2 — corrected)
------------------------------------
Generates synthetic daily OHLCV data for a basket of NSE Nifty-50
constituents. Live NSE/exchange data feeds are not reachable from this
execution environment, so price paths are simulated. Two corrections vs.
the original version, made after a self-review of the first submission:

  FIX 1 (variance recursion was dead code): the original GARCH(1,1) variance
  update multiplied the shock term by 0.0, so volatility clustering never
  actually happened -- it was plain GBM with a mislabeled comment. This
  version uses a real GARCH(1,1) recursion driven by the PREVIOUS day's
  actual return shock: var_t = omega + alpha*eps_{t-1}^2 + beta*var_{t-1}.

  FIX 2 (no cross-stock correlation): the original version drew fully
  independent shocks per ticker, which understates real portfolio risk --
  Nifty-50 large-caps move together during market-wide moves. This version
  adds a single shared "market factor" shock per day; each stock's return
  is beta*market_shock + idiosyncratic_shock, with idiosyncratic variance
  reduced so total annualised volatility still matches the target sigma
  for that stock.

All randomness is seeded for full reproducibility.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ------------------------------------------------------------------
# Universe: (ticker, sector, start price INR, annual drift, annual vol, market beta)
# ------------------------------------------------------------------
UNIVERSE = [
    ("RELIANCE",  "Energy/Retail",   2900.0, 0.12, 0.22, 1.00),
    ("TCS",       "IT Services",     4150.0, 0.10, 0.20, 0.85),
    ("HDFCBANK",  "Banking",         1720.0, 0.13, 0.21, 1.15),
    ("INFY",      "IT Services",     1900.0, 0.11, 0.23, 0.90),
    ("ICICIBANK", "Banking",         1290.0, 0.14, 0.24, 1.20),
    ("SBIN",      "Banking (PSU)",    810.0, 0.15, 0.28, 1.35),
    ("ITC",       "FMCG",             468.0, 0.09, 0.18, 0.60),
    ("LT",        "Capital Goods",   3600.0, 0.16, 0.26, 1.25),
    ("KOTAKBANK", "Banking",         1810.0, 0.10, 0.22, 1.10),
    ("HINDUNILVR","FMCG",            2480.0, 0.08, 0.17, 0.55),
]

MARKET_ANNUAL_VOL = 0.16
TRADING_DAYS_PER_YEAR = 252
N_DAYS = 504
START_DATE = "2024-01-02"


def garch11_shocks(n_days, annual_sigma, seed, alpha=0.08, beta=0.90):
    """Real GARCH(1,1): variance driven by the PREVIOUS day's realised
    shock (not zeroed out). alpha+beta < 1 for stationarity."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    long_run_var = annual_sigma ** 2 * dt
    omega = long_run_var * (1 - alpha - beta)
    omega = max(omega, long_run_var * 0.02)

    var = long_run_var
    eps_prev = 0.0
    shocks = np.zeros(n_days)
    for t in range(n_days):
        var = omega + alpha * (eps_prev ** 2) + beta * var
        var = max(var, 1e-10)
        eps = rng.standard_normal() * np.sqrt(var)
        shocks[t] = eps
        eps_prev = eps
    return shocks


def simulate_correlated_paths(universe, n_days, market_annual_vol):
    market_shocks = garch11_shocks(n_days, market_annual_vol, seed=7)
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    all_closes = {}
    for idx, (ticker, sector, s0, mu, sigma, beta) in enumerate(universe):
        idio_var_annual = max(sigma ** 2 - (beta * market_annual_vol) ** 2, (0.05 * sigma) ** 2)
        idio_sigma_annual = np.sqrt(idio_var_annual)
        idio_shocks = garch11_shocks(n_days, idio_sigma_annual, seed=100 + idx)

        total_shocks = beta * market_shocks + idio_shocks
        drift = (mu - 0.5 * sigma ** 2) * dt
        log_returns = drift + total_shocks
        closes = s0 * np.exp(np.cumsum(log_returns))
        all_closes[ticker] = closes
    return all_closes, market_shocks


def build_ohlcv(ticker, closes, seed, dates):
    rng = np.random.default_rng(seed)
    n_days = len(closes)
    opens, highs, lows, vols = [], [], [], []
    prev_close = closes[0]
    for i in range(n_days):
        c = closes[i]
        day_ret = (c / prev_close) - 1
        gap = rng.normal(0, abs(day_ret) * 0.4 + 0.002)
        o = prev_close * (1 + gap)
        intraday_range = abs(rng.normal(0, abs(day_ret))) + 0.004
        h = max(o, c) * (1 + intraday_range * rng.uniform(0.2, 0.6))
        l = min(o, c) * (1 - intraday_range * rng.uniform(0.2, 0.6))
        base_vol = rng.lognormal(mean=14.2, sigma=0.4)
        vol = int(base_vol * (1 + 3 * abs(gap)))
        opens.append(o); highs.append(h); lows.append(l); vols.append(vol)
        prev_close = c

    df = pd.DataFrame({
        "date": dates,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": vols,
    })
    df["ticker"] = ticker
    return df


def main():
    dates = pd.bdate_range(start=START_DATE, periods=N_DAYS)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    closes_by_ticker, market_shocks = simulate_correlated_paths(UNIVERSE, N_DAYS, MARKET_ANNUAL_VOL)

    market_index = 15000.0 * np.exp(np.cumsum(market_shocks))
    pd.DataFrame({"date": dates, "market_factor_index": np.round(market_index, 2)}).to_csv(
        os.path.join(out_dir, "market_factor.csv"), index=False)

    all_frames = []
    for idx, (ticker, sector, s0, mu, sigma, beta) in enumerate(UNIVERSE):
        df = build_ohlcv(ticker, closes_by_ticker[ticker], seed=200 + idx, dates=dates)
        df["sector"] = sector
        df.to_csv(os.path.join(out_dir, f"{ticker}.csv"), index=False)
        all_frames.append(df)
        print(f"Generated {ticker} (beta={beta}): {len(df)} bars, close range "
              f"{df['close'].min():.1f}-{df['close'].max():.1f}")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "nifty_universe_combined.csv"), index=False)

    close_matrix = pd.DataFrame({t: closes_by_ticker[t] for t, *_ in UNIVERSE})
    ret_matrix = np.log(close_matrix / close_matrix.shift(1)).dropna()
    avg_pairwise_corr = (ret_matrix.corr().values.sum() - len(UNIVERSE)) / (len(UNIVERSE) * (len(UNIVERSE) - 1))
    print(f"\nAverage pairwise daily-return correlation across the basket: {avg_pairwise_corr:.3f}")
    print(f"Saved {len(UNIVERSE)} tickers, {len(combined)} total rows -> {out_dir}")


if __name__ == "__main__":
    main()
