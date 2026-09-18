"""AI integration validation.
Trains a Random Forest trade-quality classifier only on the in-sample period,
then applies a frozen probability filter to held-out rule-based entry signals.
"""
import os, sys, json
import pandas as pd
sys.path.insert(0,os.path.dirname(__file__))
from data_io import load_raw_data_from_workbook
from universe import TICKERS, IN_SAMPLE_END, OUT_OF_SAMPLE_START
from ai_filter import prepare_training_frame, fit_model, evaluate_model
from strategy import generate_signals
from backtester import PortfolioBacktester
from risk_manager import RiskManager
from metrics import full_report

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA=os.path.join(BASE,'data'); LOGS=os.path.join(BASE,'logs')
BEST_STRATEGY={'ema_fast':10,'ema_slow':30,'rsi_entry_low':40,'rsi_entry_high':70}
BEST_RISK={'stop_loss_atr_mult':3.0,'take_profit_atr_mult':3.0}

# Frozen threshold: chosen ex ante as a simple probability-above-chance filter,
# not optimized on OOS data.
AI_THRESHOLD=0.50

def run():
    raw=load_raw_data_from_workbook(TICKERS)
    train=prepare_training_frame(raw,BEST_STRATEGY,end_date=IN_SAMPLE_END)
    model=fit_model(train)
    eval_rows=[]
    for t,d in raw.items():
        x=generate_signals(d.copy(),**BEST_STRATEGY)
        from ai_filter import add_ai_features
        x=add_ai_features(x)
        x=x[x['date']>=OUT_OF_SAMPLE_START].copy()
        x['ai_probability']=model.predict_proba(x[[*__import__('ai_filter').FEATURES]].fillna(0))[:,1]
        x['ai_pass']=x['ai_probability']>=AI_THRESHOLD
        eval_rows.append(x)
    scored=pd.concat(eval_rows,ignore_index=True)
    scored.to_csv(os.path.join(LOGS,'ai_oos_signal_scores.csv'),index=False)
    # Build preloaded data with AI filter applied only to entry signals.
    prepared={}
    for t,d in raw.items():
        x=generate_signals(d.copy(),**BEST_STRATEGY)
        from ai_filter import add_ai_features
        x=add_ai_features(x)
        x['ai_probability']=model.predict_proba(x[[*__import__('ai_filter').FEATURES]].fillna(0))[:,1]
        x.loc[x['date']>=OUT_OF_SAMPLE_START,'long_entry']=x.loc[x['date']>=OUT_OF_SAMPLE_START,'long_entry'] & (x.loc[x['date']>=OUT_OF_SAMPLE_START,'ai_probability']>=AI_THRESHOLD)
        prepared[t]=x.set_index('date')
    rm=RiskManager(**BEST_RISK)
    bt=PortfolioBacktester(DATA,TICKERS,rm,preloaded_data=prepared,date_range=(OUT_OF_SAMPLE_START,max(d['date'].max() for d in raw.values())))
    eq,tr=bt.run(); rep,_=full_report(eq['mark_to_market_equity'],tr)
    # Model diagnostic on held-out labeled rows.
    test_labeled=[]
    for t,d in raw.items():
        from ai_filter import add_ai_features
        x=generate_signals(d.copy(),**BEST_STRATEGY); x=add_ai_features(x)
        x['future_return']=x['close'].shift(-5)/x['close']-1; x['target']=(x['future_return']>0).astype(int)
        x=x[x['date']>=OUT_OF_SAMPLE_START]
        test_labeled.append(x)
    eval_df=pd.concat(test_labeled,ignore_index=True).dropna(subset=__import__('ai_filter').FEATURES+['target'])
    diag=evaluate_model(model,eval_df)
    summary={'ai_model':'Random Forest classifier','training_end':str(IN_SAMPLE_END.date()),'oos_start':str(OUT_OF_SAMPLE_START.date()),
             'threshold':AI_THRESHOLD,'training_rows':len(train),**diag,
             **{f'ai_filtered_oos_{k}':v for k,v in rep.items()}}
    with open(os.path.join(LOGS,'ai_oos_result.json'),'w') as f: json.dump(summary,f,indent=2,default=str)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': run()
