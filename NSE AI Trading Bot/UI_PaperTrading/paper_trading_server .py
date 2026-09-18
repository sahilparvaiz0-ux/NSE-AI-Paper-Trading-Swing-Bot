import os, sys, json, traceback
from pathlib import Path
from datetime import datetime

# This file is intended to live in:
#   <Bot Outcomes>/UI_PaperTrading/paper_trading_server.py
# It automatically finds the submitted project one level above.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = None
for p in [HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent]:
    if (p / "src").is_dir():
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError("Could not find the submitted project root containing the src folder.")

SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from backtester import PortfolioBacktester
from risk_manager import RiskManager
from strategy import generate_signals
from universe import TICKERS, IN_SAMPLE_END
from metrics import full_report
from ai_filter import add_ai_features, prepare_training_frame, fit_model

UI_FILE = HERE / "NSE_AI_TradingBot_Connected_UI.html"
STATE_FILE = HERE / "paper_trading_state.json"

# The submitted data_io.py expects data/nifty_data.xlsx, while the supplied
# workbook may be placed at project-root/nifty_data.xlsx. The bridge supports
# both locations without changing the submitted source files.
WORKBOOK_CANDIDATES = [
    PROJECT_ROOT / "nifty_data.xlsx",
    PROJECT_ROOT / "data" / "nifty_data.xlsx",
    HERE / "nifty_data.xlsx",
]
WORKBOOK = next((p for p in WORKBOOK_CANDIDATES if p.exists()), None)
if WORKBOOK is None:
    raise FileNotFoundError("nifty_data.xlsx was not found in the project root or data folder.")

app = Flask(__name__)

RAW_CACHE = None
AI_MODEL = None

def load_raw():
    global RAW_CACHE
    if RAW_CACHE is None:
        sheets = pd.read_excel(WORKBOOK, sheet_name=None)
        RAW_CACHE = {}
        for t in TICKERS:
            if t not in sheets:
                raise KeyError(f"Workbook is missing ticker sheet: {t}")
            d = sheets[t].copy()
            d = d.rename(columns={"Date":"date","Ticker":"ticker"})
            d["date"] = pd.to_datetime(d["date"])
            RAW_CACHE[t] = d.sort_values("date").reset_index(drop=True)
    return RAW_CACHE

def signal_frame(ticker):
    raw = load_raw()[ticker].copy()
    return generate_signals(raw).reset_index(drop=True)

def latest_signal(ticker):
    d = signal_frame(ticker)
    r = d.iloc[-1]
    return {
        "ticker": ticker, "date": str(pd.Timestamp(r["date"]).date()),
        "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]),
        "close": float(r["close"]), "ema_fast": float(r["ema_fast"]), "ema_slow": float(r["ema_slow"]),
        "rsi": float(r["rsi14"]), "macd_hist": float(r["macd_hist"]), "atr": float(r["atr14"]),
        "rule_entry": bool(r["long_entry"]), "rule_exit": bool(r["long_exit_signal"]),
        "signal_strength": float(r.get("signal_strength", 0.0)),
        "stop_loss": float(r["close"] - 2.0*r["atr14"]) if pd.notna(r["atr14"]) else None,
        "take_profit": float(r["close"] + 4.0*r["atr14"]) if pd.notna(r["atr14"]) else None,
    }

def get_ai_model():
    global AI_MODEL
    if AI_MODEL is None:
        train = prepare_training_frame(load_raw(), {}, horizon=5, end_date=IN_SAMPLE_END)
        AI_MODEL = fit_model(train, seed=42)
    return AI_MODEL

def enrich_ai(x):
    # Keep AI separate from rule signals. The submitted AI target is next-five-day
    # positive return; the model is trained only through IN_SAMPLE_END and frozen.
    try:
        model = get_ai_model()
        z = add_ai_features(x.copy())
        z["ai_probability"] = model.predict_proba(z[["ema_gap_pct","rsi14","macd_hist_pct","atr_pct","ret_1d","ret_5d","vol_z"]].fillna(0))[:,1]
        z["ai_pass"] = z["ai_probability"] >= 0.50
        return z
    except Exception:
        x = x.copy()
        x["ai_probability"] = np.nan
        x["ai_pass"] = np.nan
        return x

def latest_with_ai(ticker):
    d = enrich_ai(signal_frame(ticker))
    r = d.iloc[-1]
    out = latest_signal(ticker)
    out["ai_probability"] = None if pd.isna(r["ai_probability"]) else float(r["ai_probability"])
    out["ai_pass"] = None if pd.isna(r["ai_pass"]) else bool(r["ai_pass"])
    return out

def default_state():
    return {
        "starting_capital": 1000000.0,
        "cash": 1000000.0,
        "positions": {},
        "orders": [],
        "equity_history": [],
    }

def load_state():
    if not STATE_FILE.exists():
        return default_state()
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return default_state()

def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))

def market_price(ticker):
    return float(load_raw()[ticker].iloc[-1]["close"])

def current_state():
    s = load_state()
    positions = []
    market_value = 0.0
    for t,p in s["positions"].items():
        px = market_price(t)
        mv = px * p["qty"]
        upnl = (px - p["entry_price"]) * p["qty"]
        market_value += mv
        positions.append({
            "ticker":t, "qty":p["qty"], "entry_price":p["entry_price"],
            "current_price":px, "market_value":mv, "unrealized_pnl":upnl
        })
    equity = s["cash"] + market_value
    return s, positions, equity

@app.get("/")
def index():
    return send_from_directory(HERE, UI_FILE.name)

@app.get("/api/state")
def api_state():
    s, positions, equity = current_state()
    signals = [latest_with_ai(t) for t in TICKERS]
    last_date = max(x["date"] for x in signals)
    return jsonify({
        "starting_capital":s["starting_capital"], "cash":s["cash"], "equity":equity,
        "pnl":equity-s["starting_capital"], "positions":positions, "orders":s["orders"],
        "equity_history":s["equity_history"], "signals":signals,
        "last_market_date":last_date, "data_source":str(WORKBOOK),
        "paper_only":True
    })

@app.get("/api/scan")
def api_scan():
    return jsonify({"signals":[latest_with_ai(t) for t in TICKERS]})

@app.get("/api/analyze/<ticker>")
def api_analyze(ticker):
    t = ticker.upper()
    if t not in TICKERS: return jsonify({"error":"Ticker not in submitted universe"}),400
    return jsonify(latest_with_ai(t))

@app.post("/api/order")
def api_order():
    body=request.get_json(force=True)
    side=str(body.get("side","")).upper()
    t=str(body.get("ticker","")).upper()
    qty=int(body.get("qty",0))
    if side not in {"BUY","SELL"} or t not in TICKERS or qty<=0:
        return jsonify({"error":"Use BUY/SELL, a submitted ticker, and positive integer quantity."}),400
    s=load_state(); px=market_price(t); now=datetime.now().isoformat(timespec="seconds")
    if side=="BUY":
        cost=px*qty
        if cost>s["cash"]: return jsonify({"error":f"Insufficient paper cash. Need ₹{cost:,.2f}, have ₹{s['cash']:,.2f}."}),400
        p=s["positions"].setdefault(t,{"qty":0,"entry_price":px})
        new_qty=p["qty"]+qty
        p["entry_price"]=((p["qty"]*p["entry_price"])+(qty*px))/new_qty
        p["qty"]=new_qty
        s["cash"]-=cost
    else:
        if t not in s["positions"] or s["positions"][t]["qty"]<qty:
            return jsonify({"error":"Not enough paper position to sell."}),400
        s["cash"]+=px*qty
        s["positions"][t]["qty"]-=qty
        if s["positions"][t]["qty"]==0: del s["positions"][t]
    s["orders"].append({"timestamp":now,"side":side,"ticker":t,"qty":qty,"price":px,"value":px*qty,"status":"PAPER_FILLED"})
    save_state(s)
    return jsonify({"ok":True,"side":side,"ticker":t,"qty":qty,"price":px})

@app.post("/api/reset")
def api_reset():
    save_state(default_state())
    return jsonify({"ok":True})

@app.get("/api/backtest")
def api_backtest():
    raw=load_raw()
    rm=RiskManager()
    bt=PortfolioBacktester(str(PROJECT_ROOT / "data"), TICKERS, rm, raw_data=raw)
    equity,trades=bt.run()
    report,_=full_report(equity["mark_to_market_equity"],trades)
    return jsonify(report)

@app.post("/api/chat")
def api_chat():
    c=str(request.get_json(force=True).get("command","")).strip()
    u=c.upper()
    try:
        if u=="HELP":
            text="Commands: ANALYZE <TICKER>, SCAN, PORTFOLIO, POSITIONS, PERFORMANCE, BUY <TICKER> <QTY>, SELL <TICKER> <QTY>"
        elif u.startswith("ANALYZE "):
            t=u.split()[1]; x=latest_with_ai(t)
            text=(f"{t} | {x['date']} | Close ₹{x['close']:,.2f}\n"
                  f"EMA20 ₹{x['ema_fast']:,.2f} | EMA50 ₹{x['ema_slow']:,.2f} | RSI {x['rsi']:.2f} | MACD Hist {x['macd_hist']:.4f}\n"
                  f"Rule entry: {x['rule_entry']} | Rule exit: {x['rule_exit']}\n"
                  f"AI probability: {x['ai_probability'] if x['ai_probability'] is not None else 'N/A'} | AI pass: {x['ai_pass']}\n"
                  f"Stop: ₹{x['stop_loss']:,.2f} | Target: ₹{x['take_profit']:,.2f}")
        elif u=="SCAN":
            xs=[x for x in [latest_with_ai(t) for t in TICKERS] if x["rule_entry"]]
            text="Rule-based entry signals: "+(", ".join(x["ticker"] for x in xs) if xs else "None on latest available date.")
        elif u in {"PORTFOLIO","POSITIONS"}:
            s,p,e=current_state()
            text=f"Cash ₹{s['cash']:,.2f}\nEquity ₹{e:,.2f}\nP&L ₹{e-s['starting_capital']:,.2f}\nPositions: "+(", ".join(f"{x['ticker']} {x['qty']}" for x in p) if p else "None")
        elif u=="PERFORMANCE":
            # Read the submitted performance summary when available; otherwise run it.
            p=PROJECT_ROOT/"logs"/"performance_summary.json"
            if p.exists():
                r=json.loads(p.read_text())
                text=(f"Submitted backtest: return {r.get('total_return_pct',0):.2f}%, CAGR {r.get('cagr_pct',0):.2f}%, "
                      f"Sharpe {r.get('sharpe_ratio',0):.2f}, max drawdown {r.get('max_drawdown_pct',0):.2f}%, "
                      f"trades {r.get('total_trades',0)}.")
            else:
                text="Run the Backtest tab to calculate performance from the submitted PortfolioBacktester."
        elif u.startswith("BUY ") or u.startswith("SELL "):
            parts=u.split(); side=parts[0]; t=parts[1]; q=int(parts[2])
            with app.test_request_context("/api/order",method="POST",json={"side":side,"ticker":t,"qty":q}):
                resp=api_order()
                if isinstance(resp,tuple): return jsonify({"text":resp[0].get_json()["error"]})
                x=resp.get_json()
            text=f"Paper order filled: {x['side']} {x['qty']} {x['ticker']} @ ₹{x['price']:,.2f}"
        else:
            text="Unknown command. Type HELP."
        return jsonify({"text":text})
    except Exception as e:
        return jsonify({"text":str(e)}),400

if __name__=="__main__":
    print("Project root:",PROJECT_ROOT)
    print("Workbook:",WORKBOOK)
    print("UI:",UI_FILE)
    print("Paper state:",STATE_FILE)
    app.run(host="127.0.0.1",port=5000,debug=False)
