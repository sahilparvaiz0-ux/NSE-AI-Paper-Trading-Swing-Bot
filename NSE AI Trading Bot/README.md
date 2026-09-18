# NSE Nifty-50 Swing Trading Bot with AI Trade-Quality Validation

**Financial and Risk Analytics — Individual Assignment**  
**Sahil Parveez | MBA Business Analytics & AI | CHRIST (Deemed to be) University, Delhi NCR**

## 1. Project Overview

This project develops and evaluates a long-only swing-trading bot for selected NSE Nifty-50 constituents using historical daily OHLCV data.

The bot combines:

- EMA trend crossover
- RSI momentum confirmation
- MACD histogram confirmation
- ATR-based position sizing and risk controls
- Stop-loss and take-profit rules
- Transaction-cost modelling
- Explicit cash accounting
- Chronological out-of-sample validation
- A Random Forest AI trade-quality filter

The historical evaluation covers **2020-10-01 to 2025-09-30** using the supplied historical data workbook.

> **Important:** This is an academic/research trading system. It does not place live orders or connect to a brokerage account.

## 2. Strategy Logic

### Entry

A long signal is generated when:

- EMA fast crosses above EMA slow
- RSI(14) is between 40 and 70
- MACD histogram is greater than 0

Signals are detected using the **day T close** and executed at the **day T+1 open**.

### Exit

A position can exit through:

- ATR-based stop-loss
- ATR-based take-profit
- EMA cross-down
- RSI above 80

Close-based exit signals are executed at the following trading day's open.

### Risk Management

- Starting capital: Rs 10,00,000
- Risk per trade: 1% of portfolio equity
- Maximum concurrent positions: 5
- Maximum single-position allocation: 25% of equity
- Stop-loss: 2 × ATR(14)
- Take-profit: 4 × ATR(14)
- Modelled brokerage: 0.03% per trade leg
- Modelled slippage: 0.05% per trade leg
- Modelled STT: 0.10% on the sell leg
- Modelled idle-cash rate: 6.5% annually

## 3. Five-Year Baseline Results

| Metric | Result |
|---|---:|
| Starting capital | Rs 10,00,000 |
| Ending equity | Rs 15,34,922 |
| Total return | 53.49% |
| CAGR | 8.95% |
| Annualised volatility | 6.94% |
| Sharpe ratio | 0.35 |
| Sortino ratio | 0.51 |
| Maximum drawdown | -7.74% |
| Closed trades | 98 |
| Win rate | 44.0% |
| Profit factor | 1.36 |
| Expectancy | Rs 2,349/trade |

The five-year equity gain is approximately Rs 5.35 lakh. Approximately Rs 2.30 lakh came from net trading P&L and Rs 3.05 lakh from modelled interest on idle cash. Average capital deployment was 28.1%.

## 4. Risk Analytics

Historical portfolio risk measures include:

- 95% VaR: -0.62%
- 95% CVaR: -1.04%
- 99% VaR: -1.23%
- 99% CVaR: -1.68%
- Worst observed daily return: -3.13%
- Longest underwater period: 204 trading days

A stationary block bootstrap with 2,000 resamples and a mean block length of 35 trading days produced a 90% diagnostic interval of **[-0.30, 1.01]** for the Sharpe ratio.

## 5. Benchmark and Exposure Analysis

The project includes comparisons with:

- Nifty 50 price index endpoint return: 116.27%
- Daily-rebalanced equal-weight 10-stock basket: 117.89%
- Volatility-matched passive basket/cash control: 74.29%

The 129.32% figure in the analysis is the arithmetic average of the ten constituents' individual cumulative price returns and is retained only as a constituent-level diagnostic.

The Nifty 50 comparison is based on a price index rather than a total-return index, and the stock-basket calculations exclude dividends.

## 6. Out-of-Sample Validation

The validation uses one chronological train/test split:

- **In-sample:** 2020-10-01 to 2024-09-30
- **Held-out:** 2024-10-01 to 2025-09-30

Twenty parameter combinations were tested inside the in-sample period.

The selected held-out configuration was:

- EMA(10/30)
- RSI[40,70]
- 3 × ATR stop
- 3 × ATR target

Held-out rule-based results:

- Total return: **-8.48%**
- Sharpe ratio: **-2.33**
- Trades: **41**
- Win rate: **21.95%**

This is a single chronological train/test split, not rolling walk-forward re-optimisation.

## 7. AI Trade-Quality Layer

The AI layer uses a **Random Forest classifier** as a separate trade-quality filter.

Features include:

- EMA spread
- RSI
- MACD histogram relative to price
- ATR relative to price
- One-day return
- Five-day return
- Volume z-score

### Target

The target is:

**1 = closing price five trading days after the signal date is above the signal-date close; otherwise 0.**

The model is trained only through **2024-09-30**, frozen before held-out testing, and uses an ex-ante probability threshold of **0.50**.

### AI Results

- OOS accuracy: **50.96%**
- OOS AUC: **0.516**
- AI-filtered OOS return: **-4.68%**
- AI-filtered OOS Sharpe: **-2.04**
- AI-filtered trades: **34**
- AI-filtered profit factor: **0.43**

The AI layer demonstrates integration and a measurable filtering effect, but the AUC of 0.516 does not establish predictive power.

## 8. Project Structure

```text
NSE-AI-Trading-Bot/
│
├── src/
│   ├── advanced_analysis.py
│   ├── ai_filter.py
│   ├── ai_oos_validation.py
│   ├── audit_check.py
│   ├── backtester.py
│   ├── consolidate_data.py
│   ├── data_generator.py
│   ├── data_io.py
│   ├── indicators.py
│   ├── main.py
│   ├── metrics.py
│   ├── plot_benchmark_comparison.py
│   ├── real_data_loader.py
│   ├── risk_manager.py
│   ├── sensitivity_analysis.py
│   ├── strategy.py
│   ├── universe.py
│   └── walk_forward.py
│
├── nifty_data.xlsx
├── charts/
├── logs/
│
├── UI_PaperTrading/
│   ├── NSE_AI_TradingBot_Connected_UI.html
│   ├── paper_trading_server.py
│   ├── start_paper_trading.bat
│   └── paper_trading_state.json   # generated locally; not committed
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Running the Original Backtest

From the repository root:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

If your environment requires the source directory on the Python path:

```bash
set PYTHONPATH=src
python src/main.py
```

## 10. Running the Connected UI and Paper Trading

The UI is an additional layer and does not replace the original bot files.

Open a terminal in the repository root and run:

```bash
pip install -r requirements.txt
python UI_PaperTrading/paper_trading_server.py
```

Then open:

```text
http://127.0.0.1:5000
```

Alternatively, on Windows, double-click:

```text
UI_PaperTrading/start_paper_trading.bat
```

The UI can:

- Display the bot dashboard
- Scan the configured universe
- Analyse individual tickers
- Show strategy signals
- Run the backtest
- Maintain a simulated paper portfolio
- Record simulated paper orders
- Reset the paper-trading state
- Provide bot-related chat responses

The paper-trading interface is **simulation only**. It does not send orders to NSE, a broker, or a trading account.

## 11. Reproducibility

The project uses the supplied `nifty_data.xlsx` historical dataset as the data snapshot for the main analysis.

The code separates:

- Data loading
- Indicator calculation
- Strategy signal generation
- Risk management
- Backtesting
- Performance metrics
- Sensitivity analysis
- Out-of-sample validation
- AI filtering
- Advanced risk diagnostics

## 12. Limitations

Key limitations include:

- The strategy is evaluated on a fixed basket of ten Nifty-50 constituents.
- Daily OHLCV data cannot reproduce exact intraday execution.
- Stop-loss and take-profit triggers are inferred from daily high/low values.
- Overnight gaps may cause live execution to differ from the simulated trigger price.
- The fixed stock basket introduces survivorship/selection limitations.
- Parameter selection introduces data-snooping risk.
- The validation is one chronological train/test split rather than rolling walk-forward optimisation.
- The AI model has weak held-out AUC and does not establish predictive superiority.
- Historical backtest performance should not be interpreted as evidence of a persistent future trading edge.

## 13. Academic Purpose

This repository is intended to demonstrate an auditable implementation of:

**Trading Strategy → Risk Management → Backtesting → Benchmarking → Out-of-Sample Validation → AI Validation → Paper Trading Interface**

It is an academic Financial Risk Analytics project rather than a live investment system.
