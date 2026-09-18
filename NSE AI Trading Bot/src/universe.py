"""
universe.py
------------
Ticker list and sector labels for the 10-stock Nifty-50 basket used
throughout the pipeline. Kept separate from data_generator.py (now legacy)
and real_data_loader.py (the default data source) so both can import the
same universe definition without coupling to either data source.
"""

TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "ITC", "LT", "KOTAKBANK", "HINDUNILVR",
]

SECTOR_MAP = {
    "RELIANCE": "Energy/Retail",
    "TCS": "IT Services",
    "HDFCBANK": "Banking",
    "INFY": "IT Services",
    "ICICIBANK": "Banking",
    "SBIN": "Banking (PSU)",
    "ITC": "FMCG",
    "LT": "Capital Goods",
    "KOTAKBANK": "Banking",
    "HINDUNILVR": "FMCG",
}

# Shared train/test split boundary used by BOTH sensitivity_analysis.py and
# walk_forward.py, so the two stay consistent: the sensitivity grid (which
# functions as a robustness check on parameter choice) runs ONLY on the
# in-sample window, and never touches the dates walk_forward.py holds out
# as out-of-sample. Previously these overlapped, which let the "robustness
# check" section implicitly use data the "out-of-sample" section claimed
# was never touched during parameter selection.
import pandas as pd
IN_SAMPLE_END = pd.Timestamp("2024-09-30")
OUT_OF_SAMPLE_START = pd.Timestamp("2024-10-01")
