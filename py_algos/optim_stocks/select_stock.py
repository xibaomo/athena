import os,sys
import pandas as pd
import numpy as np
import pdb
import yfinance as yf
from scipy.stats import ks_2samp
from kalman_tracker import KalmanTracker
from download import DATA_FILE
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

def compute_trading_value(df,lookback,target_date=-1):
    # pdb.set_trace()
    new_df = (df['High']+df['Low']+df['Close'])/3*df['Volume']*1e-9
    new_df = new_df.iloc[-lookback:,:]

    tv = new_df.sum()
    rtn = df['Close'].pct_change()
    rtn = rtn.iloc[-lookback:,:].sum()

    dff = tv.to_frame(name='tv')
    dff['rtn'] = rtn
    dff['close'] = df['Close'].iloc[-1,:]
    return dff
def cal_profit(v,prices,N=50,cap = 10000):
    # pdb.set_trace()
    # v = xs[-N:,1]
    v = v[-N:]
    price = prices[-N:]
    transactions=[]
    is_pos_on = False
    p0=0
    c0 = cap
    # pdb.set_trace()
    for i in range(1,N):
        # if  s[i] > s[i-1] and not is_pos_on and abs(xs[i,-1]-k_eq) < 0.02:
        if v[i] > 0 and v[i] > v[i-1] and not is_pos_on:
            p0 = price[i]
            is_pos_on = True
            trans = [i,-1,-1]
            transactions.append(trans)
        if not is_pos_on:
            continue
        if  v[i] < 0:
            is_pos_on = False
            cap = cap / p0 * price[i]
            transactions[-1][1] = i
            transactions[-1][2] = price[i]/p0 - 1
    if is_pos_on:
        is_pos_on = False
        cap = cap / p0 * price[-1]
        transactions[-1][1] = N-1
        transactions[-1][2] = price[-1]/p0-1
        if transactions[-1][0] == transactions[-1][1]:
            transactions.pop()

    if len(transactions)> 0:
        return cap-c0, (cap-c0)/len(transactions)
        # print(transactions)
        # print(f"Protift -- total: ${cap-c0:.2f}, ave: ${(cap-c0)/len(transactions):.2f}")

    return 0.,0.

def kalman_addon(data,df):
    ktcal = KalmanTracker()
    df_close = data['Close']
    for ticker in df_close.keys():
        p = df_close[ticker].values
        vs,acc = ktcal.estimateMotion(np.log(p))
        df.loc[ticker,'v'] = vs[-1]
        df.loc[ticker,'acc'] = acc
        prof,ave_prof = cal_profit(vs,p)
        df.loc[ticker,'prof'] = prof
    return df

def rtn_per_risk(data,df):
    df_close = data['Close']
    dff = df_close.pct_change()

    for ticker in dff.keys():
        r = dff[ticker].values[-200:]
        risk = np.std(r)
        rpr = np.mean(r)/risk
        df.loc[ticker,'rpr'] = rpr
        df.loc[ticker,'risk'] = risk

    return df
def tv_weighted_rtn(df):

    lb = np.log(np.min(df['tv'].values))
    ub = np.log(np.max(df['tv'].values))
    for ticker in df.index.tolist():
        score = (np.log(df.loc[ticker,'tv']) - lb)/(ub-lb)
        df.loc[ticker,'tv_rtn'] = df.loc[ticker,'rpr']*score

    return df

def compute_rsi(data,df):
    df_close = data['Close']
    dff = df_close.pct_change()

    for ticker in df.index.tolist():
        r = dff[ticker].values[-60:]
        up = r[r>0].sum()
        dw = abs(r[r<0].sum())
        rsi = up/(up+dw)
        df.loc[ticker,'rsi'] = rsi
    return df

def compute_ls_rtn(data,df):
    df_close = data['Close']
    dff = df_close.pct_change()
    for ticker in df.index.tolist():
        lr = dff[ticker].values[-70:].mean()
        sr = dff[ticker].values[-20:].mean()
        if lr > 0 and sr > 0:
            df.loc[ticker,'ls_rtn'] = sr/lr
    return df
def ave_ks(ts,spacing):
    rem = len(ts) % spacing
    if rem > 0:
        ts = ts[rem:]
    n_intervals = len(ts) // spacing

    diffs = []
    for i in range(n_intervals - 1):
        seg1 = ts[i * spacing:(i + 1) * spacing]
        seg2 = ts[(i + 1) * spacing:(i + 2) * spacing]
        ks_stat, _ = ks_2samp(seg1, seg2)
        diffs.append(ks_stat)

    return np.mean(diffs) if diffs else 0.0
def df2rtns(data,ticker):
    # breakpoint()
    df_close = data['Close']
    df_high = data['High']
    df_low = data['Low']
    nrows = min(len(df_close), len(df_high), len(df_low))
    pcs = np.zeros(nrows*2)
    k=0
    for i in range(nrows):
        pcs[k] = (df_high.iloc[i][ticker] + df_low.iloc[i][ticker])/2.0
        k+=1
        pcs[k] = df_close.iloc[i][ticker]
        k+=1

    rtns = np.diff(np.log(pcs))
    rtns = rtns[~np.isnan(rtns)]
    return rtns

def ave_dist_diff(df):
    data  = yf.download(df.index.tolist(), period='730d',interval='1h')
    spacing = 22*14
    for ticker in df.index.tolist():
        rtns = df2rtns(data,ticker)
        ks = ave_ks(rtns,spacing)
        df.loc[ticker,'ks'] = ks
        print(f"ticker: {ticker}, ks: {ks}")
    return df




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_date>")
        sys.exit(1)


    target_date = sys.argv[1]  # portconf.getTargetDate()
    data = pd.read_csv(DATA_FILE, comment='#', header=[0, 1], parse_dates=[0], index_col=0)
    data = data.dropna(axis=1)
    # breakpoint()
    df = compute_trading_value(data,lookback=200)
    sorted_df = df.sort_values(by='tv', ascending=True, ignore_index=False)
    # df = sorted_df.iloc[-100:,:]
    # g = df['rtn'].values/df['tv'].values
    # df['growth'] = g
    # df = df.sort_values(by='growth', ascending=True, ignore_index=False)

    # df = kalman_addon(data,df)
    # df = df.loc[~((df[['v', 'acc', 'prof','rtn']] < 0).any(axis=1))]
    # breakpoint()
    df = rtn_per_risk(data,df)

    # breakpoint()
    df = df.sort_values(by='rpr', ascending=True, ignore_index=False)
    df = df[df['tv']>=1]
    df = df[df['rpr']>=0]

    df = tv_weighted_rtn(df)

    # breakpoint()
    base_rtns = data['Close']['SPY'].pct_change().values[-200:]
    base_risk = np.std(base_rtns)
    # df = df[df['risk']>=base_risk]
    # df = df[df['rpr']>=base_rtns.mean()/base_risk]

    # df = compute_rsi(data,df)
    # breakpoint()
    # df = compute_ls_rtn(data,df)
    # df = df[~np.isnan(df['ls_rtn']) & (df['ls_rtn']>0.5)]

    df = ave_dist_diff(df)