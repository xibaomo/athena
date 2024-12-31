import os,sys
import pandas as pd
import numpy as np
import pdb
from kalman_tracker import KalmanTracker
from download import DATA_FILE

def compute_trading_value(df,lookback=25,target_date=-1):
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} <target_date>")
        sys.exit(1)


    target_date = sys.argv[1]  # portconf.getTargetDate()
    data = pd.read_csv(DATA_FILE, comment='#', header=[0, 1], parse_dates=[0], index_col=0)
    data = data.dropna(axis=1)
    df = compute_trading_value(data)
    sorted_df = df.sort_values(by='tv', ascending=True, ignore_index=False)
    # df = sorted_df.iloc[-100:,:]
    g = df['rtn'].values/df['tv'].values
    df['growth'] = g
    df = df.sort_values(by='growth', ascending=True, ignore_index=False)

    df = kalman_addon(data,df)
    df = df.loc[~((df[['v', 'acc', 'prof','rtn']] < 0).any(axis=1))]

    df = df.sort_values(by='prof', ascending=True, ignore_index=False)