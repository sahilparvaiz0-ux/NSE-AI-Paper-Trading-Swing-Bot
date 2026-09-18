"""
data_io.py
-----------
Shared data-loading helper for the CONSOLIDATED single-workbook data
source (data/nifty_data.xlsx), which replaced the earlier per-ticker CSV
files.

Reading an XLSX workbook is noticeably slower per-call than reading a
small CSV (openpyxl has to parse the whole zipped XML structure), so this
module loads the workbook ONCE per script run, and callers that need the
same data across many backtests (e.g. every combination in a parameter
grid) should load it once and pass the resulting dict around rather than
re-reading from disk per combination. Every script in this pipeline
(main.py, sensitivity_analysis.py, walk_forward.py) follows that pattern.
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
WORKBOOK_PATH = os.path.join(DATA_DIR, "nifty_data.xlsx")


def load_raw_data_from_workbook(tickers, workbook_path=WORKBOOK_PATH):
    """Reads every requested ticker's sheet from the single workbook in ONE
    file open, returning {ticker: raw OHLCV DataFrame} (date column not yet
    the index, no strategy signals applied -- callers apply generate_signals
    themselves, since different callers may want different strategy params
    on the same raw prices)."""
    all_sheets = pd.read_excel(workbook_path, sheet_name=None)
    out = {}
    for t in tickers:
        df = all_sheets[t].copy()
        df = df.rename(columns={"Date": "date", "Ticker": "ticker"})
        df["date"] = pd.to_datetime(df["date"])
        out[t] = df
    return out
