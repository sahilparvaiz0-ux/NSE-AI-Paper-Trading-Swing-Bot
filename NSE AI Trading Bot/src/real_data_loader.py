"""
real_data_loader.py
-------------------
Loads the same real five-year NSE daily OHLCV source used for the submitted
backtest: Kaldhen-Lepcha/Historical-Data (GitHub), whose repository
documentation states that the stock data was collected through the Upstox API.

The authoritative reported results remain tied to data/nifty_data.xlsx. This
loader is provided for reproducibility and rebuilds the ten-stock input from
the repository's Assignment-2 CSV files for the exact 2020-10-01 to
2025-09-30 window.

It does not download synthetic data and does not substitute the earlier
eod2_data source. The preserved NIFTY50_INDEX benchmark endpoint in the
submitted workbook remains the benchmark snapshot used in the report.
"""

import os
import tarfile
import urllib.request
import hashlib
import datetime
import pandas as pd

from universe import TICKERS

SOURCE_REPO_TARBALL = (
    "https://codeload.github.com/Kaldhen-Lepcha/Historical-Data/"
    "tar.gz/refs/heads/main"
)
SOURCE_ATTRIBUTION = (
    "Kaldhen-Lepcha/Historical-Data (GitHub); repository documentation "
    "states the stock data was collected through the Upstox API."
)
START_DATE = pd.Timestamp("2020-10-01")
END_DATE = pd.Timestamp("2025-09-30")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RAW_CACHE_DIR = os.path.join(BASE, "_raw_nse_cache")


def download_and_extract():
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    tmp_path = os.path.join(RAW_CACHE_DIR, "_historical_data.tar.gz")

    print(f"Downloading historical OHLCV data from {SOURCE_REPO_TARBALL} ...")
    urllib.request.urlretrieve(SOURCE_REPO_TARBALL, tmp_path)

    wanted_suffixes = {
        f"Assignment-2/{t}_2020-10-01_to_2025-09-30.csv" for t in TICKERS
    }

    with tarfile.open(tmp_path, "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if any(m.name.endswith(suffix) for suffix in wanted_suffixes)
        ]
        if len(members) != len(TICKERS):
            found = {os.path.basename(m.name) for m in members}
            missing = [
                f"{t}_2020-10-01_to_2025-09-30.csv"
                for t in TICKERS
                if f"{t}_2020-10-01_to_2025-09-30.csv" not in found
            ]
            raise FileNotFoundError(
                f"Could not find all required ticker files in the source archive. "
                f"Missing: {missing}"
            )
        tar.extractall(path=RAW_CACHE_DIR, members=members)

    os.remove(tmp_path)
    return RAW_CACHE_DIR


def find_raw_csv(raw_dir, ticker):
    filename = f"{ticker}_2020-10-01_to_2025-09-30.csv"
    for root, _, files in os.walk(raw_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def standardize_and_save(raw_dir):
    os.makedirs(DATA_DIR, exist_ok=True)
    summary = []

    for ticker in TICKERS:
        raw_path = find_raw_csv(raw_dir, ticker)
        if raw_path is None:
            raise FileNotFoundError(
                f"Could not find {ticker}_2020-10-01_to_2025-09-30.csv"
            )

        df = pd.read_csv(raw_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Source schema: timestamp, open, high, low, close, volume, oi.
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker}: missing required columns {missing}")

        df = df[required].copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.drop(columns=["timestamp"])
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.sort_values("date")
        df = df[
            (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
        ].reset_index(drop=True)
        df["ticker"] = ticker

        out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)

        summary.append(
            (
                ticker,
                df["date"].min().date(),
                df["date"].max().date(),
                len(df),
                df["close"].min(),
                df["close"].max(),
            )
        )
        print(
            f"{ticker:12s}: {df['date'].min().date()} -> "
            f"{df['date'].max().date()} ({len(df)} bars)"
        )

    combined = pd.concat(
        [pd.read_csv(os.path.join(DATA_DIR, f"{t}.csv")) for t in TICKERS],
        ignore_index=True,
    )
    combined.to_csv(
        os.path.join(DATA_DIR, "nifty_universe_combined.csv"), index=False
    )
    return summary


def consolidate_and_cleanup(summary):
    """Build the canonical workbook and remove temporary per-ticker CSVs."""
    from consolidate_data import main as build_workbook

    checksum = build_workbook()

    for t in TICKERS:
        p = os.path.join(DATA_DIR, f"{t}.csv")
        if os.path.exists(p):
            os.remove(p)

    p = os.path.join(DATA_DIR, "nifty_universe_combined.csv")
    if os.path.exists(p):
        os.remove(p)

    with open(os.path.join(DATA_DIR, "DATA_SOURCE.txt"), "w") as f:
        f.write(f"Source: {SOURCE_ATTRIBUTION}\n")
        f.write("Downloaded: real historical daily OHLCV data, not synthetic.\n")
        f.write(
            f"Window kept: {START_DATE.date()} -> {END_DATE.date()} "
            f"(five years).\n"
        )
        f.write(
            "\nThe supplied data/nifty_data.xlsx remains the authoritative "
            "snapshot for the reported results.\n"
        )
        f.write(
            "The NIFTY50_INDEX sheet is a preserved benchmark endpoint "
            "snapshot and is not required for strategy signals or AI features.\n"
        )
        f.write(
            f"\nDownload timestamp (UTC): "
            f"{datetime.datetime.utcnow().isoformat()}Z\n"
        )
        f.write(f"\nGenerated workbook SHA-256: {checksum}\n")
        f.write("\nPer-ticker window:\n")
        for t, dmin, dmax, n, cmin, cmax in summary:
            f.write(
                f"  {t:12s} {dmin} -> {dmax} ({n} rows), "
                f"close {cmin:.2f}-{cmax:.2f}\n"
            )

    return checksum


def main():
    raw_dir = download_and_extract()
    summary = standardize_and_save(raw_dir)
    consolidate_and_cleanup(summary)
    print("Real five-year data load and consolidation complete.")


if __name__ == "__main__":
    main()
