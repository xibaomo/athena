import os,sys
import pandas as pd
import pdb
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



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} <target_date>")
        sys.exit(1)


    target_date = sys.argv[1]  # portconf.getTargetDate()
    data = pd.read_csv(DATA_FILE, comment='#', header=[0, 1], parse_dates=[0], index_col=0)
    data = data.dropna(axis=1)
    df = compute_trading_value(data)
    sorted_df = df.sort_values(by='tv', ascending=True, ignore_index=False)
    df = sorted_df.iloc[-100:,:]
    g = df['rtn'].values/df['tv'].values
    df['growth'] = g
    df = df.sort_values(by='growth', ascending=True, ignore_index=False)
