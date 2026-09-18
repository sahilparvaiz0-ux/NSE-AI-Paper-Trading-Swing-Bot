"""Reproducible benchmark comparison using the supplied five-year endpoints.
The Nifty 50 benchmark is deliberately shown as a cumulative-return bar rather
than interpolated as a daily path because only the verified start/end index
levels are available in the supplied benchmark evidence.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_io import load_raw_data_from_workbook, WORKBOOK_PATH
from universe import TICKERS

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS=os.path.join(BASE,'logs'); CHARTS=os.path.join(BASE,'charts')

def proxy_return(sheets):
    closes=[]
    for t in TICKERS:
        d=sheets[t].copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date')['close']
        closes.append(d/d.iloc[0]-1)
    return float(pd.concat(closes,axis=1).mean(axis=1).iloc[-1]*100)

def main():
    equity=pd.read_csv(os.path.join(LOGS,'equity_curve.csv'),parse_dates=['date'])
    bot=(equity['mark_to_market_equity'].iloc[-1]/equity['mark_to_market_equity'].iloc[0]-1)*100
    sheets=load_raw_data_from_workbook(TICKERS)
    proxy=proxy_return(sheets)
    idx=pd.read_excel(WORKBOOK_PATH,sheet_name='NIFTY50_INDEX')
    nifty=(float(idx['close'].iloc[-1])/float(idx['close'].iloc[0])-1)*100
    out=pd.DataFrame({'benchmark':['Swing Bot','Nifty 50 Price Index','Equal-weight 10-stock proxy'],
                      'cumulative_return_pct':[bot,nifty,proxy]})
    out.to_csv(os.path.join(LOGS,'benchmark_summary.csv'),index=False)
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.bar(out['benchmark'],out['cumulative_return_pct'])
    ax.axhline(0,linewidth=.8)
    ax.set_ylabel('Cumulative return (%)')
    ax.set_title('Five-Year Cumulative Return Comparison')
    ax.tick_params(axis='x',rotation=15)
    for i,v in enumerate(out['cumulative_return_pct']): ax.text(i,v+2 if v>=0 else v-2,f'{v:.2f}%',ha='center',va='bottom' if v>=0 else 'top')
    fig.tight_layout(); fig.savefig(os.path.join(CHARTS,'08_bot_vs_nifty50_index.png'),dpi=150); plt.close(fig)
    print(out.to_string(index=False))

if __name__=='__main__': main()
