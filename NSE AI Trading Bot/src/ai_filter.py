"""Explainable ML trade-quality filter.

The model is deliberately separated from the rule-based strategy. It is trained
only on the in-sample window and then frozen for held-out testing, so future
returns from the OOS period cannot leak into training.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURES=['ema_gap_pct','rsi14','macd_hist_pct','atr_pct','ret_1d','ret_5d','vol_z']

def add_ai_features(df):
    x=df.copy()
    x['ema_gap_pct']=(x['ema_fast']/x['ema_slow']-1)*100
    x['macd_hist_pct']=x['macd_hist']/x['close']*100
    x['atr_pct']=x['atr14']/x['close']*100
    x['ret_1d']=x['close'].pct_change()
    x['ret_5d']=x['close'].pct_change(5)
    vol_mean=x['volume'].rolling(20).mean(); vol_std=x['volume'].rolling(20).std()
    x['vol_z']=(x['volume']-vol_mean)/vol_std.replace(0,np.nan)
    return x

def prepare_training_frame(raw_data, strategy_params, horizon=5, end_date=None):
    from strategy import generate_signals
    rows=[]
    for ticker,raw in raw_data.items():
        d=generate_signals(raw.copy(),**strategy_params)
        d=add_ai_features(d)
        d['future_return']=d['close'].shift(-horizon)/d['close']-1
        d['target']=(d['future_return']>0).astype(int)
        d['ticker']=ticker
        rows.append(d)
    out=pd.concat(rows,ignore_index=True)
    if end_date is not None: out=out[out['date']<=end_date]
    out=out.dropna(subset=FEATURES+['future_return'])
    return out

def fit_model(train_df, seed=42):
    model=RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=8,
                                 class_weight='balanced_subsample',random_state=seed,n_jobs=-1)
    model.fit(train_df[FEATURES],train_df['target'])
    return model

def score_frame(model, df):
    x=add_ai_features(df.copy())
    x['ai_probability']=model.predict_proba(x[FEATURES].fillna(0))[:,1]
    return x

def evaluate_model(model, eval_df):
    x=eval_df.dropna(subset=FEATURES+['target']).copy()
    p=model.predict_proba(x[FEATURES])[:,1]
    return {'accuracy':float(accuracy_score(x['target'],p>=0.5)),
            'auc':float(roc_auc_score(x['target'],p)) if x['target'].nunique()>1 else np.nan,
            'n':int(len(x))}
