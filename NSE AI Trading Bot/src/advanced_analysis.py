import os, json, shutil, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS=os.path.join(BASE,'logs'); CHARTS=os.path.join(BASE,'charts'); SRC=os.path.join(BASE,'src')
os.makedirs(LOGS,exist_ok=True); os.makedirs(CHARTS,exist_ok=True)

eq=pd.read_csv(os.path.join(LOGS,'equity_curve.csv'),parse_dates=['date']).set_index('date')
tr=pd.read_csv(os.path.join(LOGS,'trade_log.csv'),parse_dates=['entry_date','exit_date'])
# daily returns
er=eq['mark_to_market_equity'].pct_change().dropna()
rf=0.065/252
TRADING_DAYS=252
# VaR/CVaR historical on daily portfolio returns
var95=np.quantile(er,0.05); var99=np.quantile(er,0.01)
cvar95=er[er<=var95].mean(); cvar99=er[er<=var99].mean()
# drawdown duration
growth=(1+er).cumprod(); peak=growth.cummax(); dd=growth/peak-1
maxdd=float(dd.min())
# underwater days longest consecutive
under=(dd<0).astype(int); runs=[]; cur=0
for x in under:
    cur=cur+1 if x else 0
    runs.append(cur)
longest=int(max(runs)); prop=float(under.mean()*100)
# skew kurt
skew=float(er.skew()); exk=float(er.kurtosis())
# attribution
start=float(eq.mark_to_market_equity.iloc[0]); end=float(eq.mark_to_market_equity.iloc[-1]); gain=end-start
trade_pnl=float(tr.pnl.sum()); interest=gain-trade_pnl
invested=(1-eq.cash/eq.mark_to_market_equity).clip(0,1)
attr={
 'total_equity_gain_inr':gain,'net_trading_pnl_inr':trade_pnl,'interest_on_idle_cash_inr':interest,
 'trading_share_of_gain_pct':trade_pnl/gain*100,'interest_share_of_gain_pct':interest/gain*100,
 'avg_capital_deployed_pct':float(invested.mean()*100),'median_capital_deployed_pct':float(invested.median()*100),
 'pct_days_fully_in_cash':float((eq.open_positions==0).mean()*100),'avg_open_positions':float(eq.open_positions.mean()),
 'max_concurrent_positions_allowed':5,
 'historical_var_95_daily_pct':float(var95*100),'historical_cvar_95_daily_pct':float(cvar95*100),
 'historical_var_99_daily_pct':float(var99*100),'historical_cvar_99_daily_pct':float(cvar99*100),
 'worst_daily_return_pct':float(er.min()*100),'skewness':skew,'excess_kurtosis':exk,
 'longest_underwater_days':longest,'pct_days_underwater':prop
}
# market basket
xls=pd.ExcelFile(os.path.join(BASE,'data/nifty_data.xlsx'))
closes={}
for t in ['RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','SBIN','ITC','LT','KOTAKBANK','HINDUNILVR']:
 d=pd.read_excel(xls,sheet_name=t); d.columns=[c.lower() for c in d.columns]; d['date']=pd.to_datetime(d['date']); closes[t]=d.set_index('date')['close'].sort_index()
px=pd.DataFrame(closes).dropna(); basket=px.pct_change().mean(axis=1).dropna()
common=er.index.intersection(basket.index); br=basket.loc[common]; bot=er.loc[common]
# volatility matched passive blend
x=float(bot.std()/br.std()); blend=x*br+(1-x)*rf
# helper
def stats(r,label):
 r=r.dropna(); growth=(1+r).cumprod(); years=(r.index[-1]-r.index[0]).days/365.25
 return {'portfolio':label,'total_return_pct':float((growth.iloc[-1]-1)*100),'cagr_pct':float((growth.iloc[-1]**(1/years)-1)*100),'ann_volatility_pct':float(r.std()*np.sqrt(252)*100),'sharpe_ratio':float((r.mean()-rf)/r.std()*np.sqrt(252)),'max_drawdown_pct':float((growth/growth.cummax()-1).min()*100)}
comp=[stats(bot,'Trading bot'),stats(br,'100% equal-weight basket'),stats(blend,f'{x*100:.1f}% basket / {(1-x)*100:.1f}% cash (vol-matched)')]
# additional static blends and idle cash overlay
for w in [0.4,0.5]: comp.append(stats(w*br+(1-w)*rf,f'{w*100:.0f}% basket / {(1-w)*100:.0f}% cash'))
cash_w=(eq.cash/eq.mark_to_market_equity).reindex(common).shift(1).fillna(1).clip(0,1)
for sleeve in [0.5,1.0]: comp.append(stats(bot+sleeve*cash_w*(br-rf),f'Bot + {sleeve*100:.0f}% index sleeve on idle cash'))
pd.DataFrame(comp).to_csv(os.path.join(LOGS,'benchmark_comparison.csv'),index=False)
# regression excess returns
Y=(bot-rf).values; X=(br-rf).values
beta,alpha=np.polyfit(X,Y,1); resid=Y-(alpha+beta*X); n=len(X)
se_alpha=np.sqrt((resid@resid)/(n-2)*(1/n+X.mean()**2/((X-X.mean())@(X-X.mean()))))
talpha=alpha/se_alpha; r2=float(np.corrcoef(X,Y)[0,1]**2)
reg={'beta_vs_equal_weight_basket':float(beta),'alpha_daily':float(alpha),'alpha_annualised_pct':float(alpha*252*100),'alpha_t_statistic':float(talpha),'r_squared':r2,'vol_matched_basket_weight':x,'avg_pairwise_return_correlation':float(px.pct_change().corr().where(~np.eye(len(px.columns),dtype=bool)).stack().mean()),'per_ticker_buy_hold_return_pct':((px.iloc[-1]/px.iloc[0]-1)*100).round(2).to_dict(),'note':'Price returns only; dividends excluded.'}
json.dump(reg,open(os.path.join(LOGS,'market_model_regression.json'),'w'),indent=2)
px.pct_change().corr().round(3).to_csv(os.path.join(LOGS,'correlation_matrix.csv'))
# year table
y=tr.groupby(tr.exit_date.dt.year).agg(trades=('pnl','size'),net_trading_pnl_inr=('pnl','sum'),win_rate_pct=('pnl',lambda s:(s>0).mean()*100))
eqy=eq.mark_to_market_equity.groupby(eq.index.year)
y['interest_inr']=0.0
y['equity_change_pct']=0.0
for yr in y.index:
 first=eqy.get_group(yr).iloc[0]; last=eqy.get_group(yr).iloc[-1]
 pnl=float(tr.loc[tr.exit_date.dt.year==yr,'pnl'].sum()); y.loc[yr,'equity_change_pct']=(last/first-1)*100; y.loc[yr,'interest_inr']=(last-first)-pnl
y.round(2).to_csv(os.path.join(LOGS,'regime_year_table.csv'))
# cost decomposition from V2 trade log: buy brokerage+slippage, sell brokerage+slippage+STT
buy_notional=(tr.entry_price*tr.qty); sell_notional=(tr.exit_price*tr.qty)
broker_buy=buy_notional*0.0003; slip_buy=buy_notional*0.0005
broker_sell=sell_notional*0.0003; slip_sell=sell_notional*0.0005; stt=sell_notional*0.001
costs={'brokerage_inr':float((broker_buy+broker_sell).sum()),'slippage_inr':float((slip_buy+slip_sell).sum()),'stt_inr':float(stt.sum())}
costs['total_transaction_cost_inr']=sum(costs.values()); costs['gross_pnl_inr']=float(tr.gross_pnl.sum()); costs['net_trading_pnl_inr']=trade_pnl; costs['costs_as_pct_of_gross_pnl']=costs['total_transaction_cost_inr']/costs['gross_pnl_inr']*100
json.dump(costs,open(os.path.join(LOGS,'cost_decomposition.json'),'w'),indent=2)
# stationary block bootstrap on daily returns; block length ~ sqrt(n)
rng=np.random.default_rng(42); vals=er.values; N=len(vals); B=2000; L=max(5,int(round(np.sqrt(N))))
sh=[]; pn=[]
for _ in range(B):
 sample=[]
 while len(sample)<N:
  start_i=rng.integers(0,N); length=rng.geometric(1/L)
  for j in range(length):
   sample.append(vals[(start_i+j)%N])
   if len(sample)>=N: break
 sample=np.array(sample[:N]); sd=sample.std(ddof=1)
 sh.append(((sample.mean()-rf)/sd*np.sqrt(252)) if sd>0 else 0); pn.append((1+sample).prod()*start-start)
block={'method':'stationary block bootstrap','n_bootstrap_samples':B,'block_length':L,'confidence_level':0.90,'sharpe_ci_low':float(np.quantile(sh,.05)),'sharpe_ci_high':float(np.quantile(sh,.95)),'total_pnl_ci_low_inr':float(np.quantile(pn,.05)),'total_pnl_ci_high_inr':float(np.quantile(pn,.95)),'probability_sharpe_leq_zero':float(np.mean(np.array(sh)<=0))}
json.dump(block,open(os.path.join(LOGS,'block_bootstrap.json'),'w'),indent=2)
# random entry null model using actual holding-period distribution and approximate notional distribution
# derive notional from entry price*qty, random ticker/date/hold/notional; same cost model as bot
N_SIMS=2000; rng=np.random.default_rng(7); holds=tr.holding_days.clip(lower=1).to_numpy(); notionals=buy_notional.to_numpy(); sims=[]
opens={t: pd.read_excel(xls,sheet_name=t).rename(columns=lambda c:c.lower()).assign(date=lambda d:pd.to_datetime(d.date)).set_index('date').open.sort_index() for t in closes}
for s in range(N_SIMS):
 total=0.0
 for k in range(len(tr)):
  t=rng.choice(list(opens)); idx=opens[t].index; hold=int(rng.choice(holds)); bars=max(int(round(hold*5/7)),1)
  if len(idx)<=bars+1: continue
  i=rng.integers(0,len(idx)-bars-1); entry=float(opens[t].iloc[i]); exitp=float(opens[t].iloc[i+bars]); notional=float(rng.choice(notionals)); qty=int(notional/entry)
  if qty<=0: continue
  gross=(exitp-entry)*qty; buy=qty*entry*(0.0003+0.0005); sell=qty*exitp*(0.0003+0.0005+0.001); total += gross-buy-sell
 sims.append(total)
sims=np.array(sims); actual=trade_pnl
null={'n_simulations':N_SIMS,'n_trades_per_simulation':len(tr),'actual_strategy_net_pnl_inr':actual,'null_mean_pnl_inr':float(sims.mean()),'null_median_pnl_inr':float(np.median(sims)),'null_5th_pct_inr':float(np.quantile(sims,.05)),'null_95th_pct_inr':float(np.quantile(sims,.95)),'actual_percentile_within_null':float(np.mean(sims<actual)*100),'one_sided_p_value':float(np.mean(sims>=actual)),'share_of_random_runs_profitable_pct':float(np.mean(sims>0)*100)}
json.dump(null,open(os.path.join(LOGS,'monte_carlo_null.json'),'w'),indent=2); pd.Series(sims,name='simulated_net_pnl_inr').to_csv(os.path.join(LOGS,'monte_carlo_null_draws.csv'),index=False)
# multiple testing summary from V2 sensitivity
sens=pd.read_csv(os.path.join(LOGS,'sensitivity_analysis.csv')); best=sens.loc[sens.sharpe_ratio.idxmax()]
rng2=np.random.default_rng(123); # null max Sharpe via resampling daily returns in-sample
# use observed grid mean/sd and iid normal null approximation, as V1 methodology; report as diagnostic only
mu=float(sens.sharpe_ratio.mean()); sd=float(sens.sharpe_ratio.std(ddof=1)); expected_max=float(mu+sd*np.sqrt(2*np.log(len(sens))))
mt={'configurations_tested':int(len(sens)),'grid_sharpe_mean':mu,'grid_sharpe_sd':sd,'observed_best_sharpe':float(best.sharpe_ratio),'expected_best_sharpe_approx_null':expected_max,'winner_distance_above_grid_mean_sd':float((best.sharpe_ratio-mu)/sd)}
json.dump(mt,open(os.path.join(LOGS,'multiple_testing_adjustment.json'),'w'),indent=2)
# charts new, no forced colors
plt.figure(figsize=(9,4.5)); plt.hist(er*100,bins=40); plt.axvline(var95*100,linestyle='--',label='95% VaR'); plt.axvline(cvar95*100,linestyle=':',label='95% CVaR'); plt.xlabel('Daily portfolio return (%)'); plt.ylabel('Frequency'); plt.title('Daily Return Distribution and Historical Tail Risk'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(CHARTS,'13_var_distribution.png'),dpi=160); plt.close()
plt.figure(figsize=(9,4.5)); vals2=[trade_pnl,interest]; plt.bar(['Net trading P&L','Idle-cash interest'],vals2); plt.ylabel('INR'); plt.title('Five-Year Return Attribution'); plt.tight_layout(); plt.savefig(os.path.join(CHARTS,'08_return_attribution.png'),dpi=160); plt.close()
plt.figure(figsize=(9,4.5)); plt.plot(eq.index,invested*100); plt.axhline(invested.mean()*100,linestyle='--',label=f'Average {invested.mean()*100:.1f}%'); plt.ylabel('Capital invested (%)'); plt.xlabel('Date'); plt.title('Capital Deployment Over the Backtest'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(CHARTS,'09_capital_deployment.png'),dpi=160); plt.close()
plt.figure(figsize=(9,4.5)); names=[c['portfolio'] for c in comp]; cagr=[c['cagr_pct'] for c in comp]; plt.barh(names,cagr); plt.xlabel('CAGR (%)'); plt.title('Trading Bot vs Passive Risk/Return Controls'); plt.tight_layout(); plt.savefig(os.path.join(CHARTS,'10_risk_return_map.png'),dpi=160); plt.close()
plt.figure(figsize=(9,4.5)); plt.hist(sims,bins=45); plt.axvline(actual,linestyle='--',label=f'Actual net P&L: Rs {actual:,.0f}'); plt.xlabel('Simulated net trading P&L (INR)'); plt.ylabel('Frequency'); plt.title('Random-Entry Null Model'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(CHARTS,'11_null_distribution.png'),dpi=160); plt.close()
plt.figure(figsize=(9,4.5)); y2=y.net_trading_pnl_inr; plt.bar(y.index.astype(str),y2); plt.axhline(0,linewidth=0.8); plt.xlabel('Year'); plt.ylabel('Net trading P&L (INR)'); plt.title('Net Trading P&L by Calendar Year'); plt.tight_layout(); plt.savefig(os.path.join(CHARTS,'12_pnl_by_year.png'),dpi=160); plt.close()
plt.figure(figsize=(8,6)); plt.imshow(px.pct_change().corr(),aspect='auto'); plt.colorbar(label='Correlation'); plt.xticks(range(len(px.columns)),px.columns,rotation=45,ha='right',fontsize=8); plt.yticks(range(len(px.columns)),px.columns,fontsize=8); plt.title('Daily Return Correlation — 10-Stock Basket'); plt.tight_layout(); plt.savefig(os.path.join(CHARTS,'14_correlation_heatmap.png'),dpi=160); plt.close()
# integrated summary
summary={'risk_analytics':attr,'benchmark_controls':comp,'market_model':reg,'cost_decomposition':costs,'block_bootstrap':block,'random_entry_null':null,'multiple_testing':mt,'year_table':y.reset_index().to_dict(orient='records')}
json.dump(summary,open(os.path.join(LOGS,'integrated_analysis_summary.json'),'w'),indent=2,default=float)
print(json.dumps({'attr':attr,'reg':reg,'null':null,'block':block,'costs':costs,'mt':mt},indent=2,default=float))
