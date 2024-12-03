import pdb

import numpy as np
from datetime import datetime, timedelta,date
import sys,os
import yfinance as yf
import matplotlib.pyplot as plt
from kalman_motion import add_days_to_date
from pick_opt import plot_double_y_axis
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from download import DATA_FILE

def comp_rtn_per_vol(df, win=10):
    rpv = np.zeros(len(df)-win)
    op = df['Open'].values
    cl = df['Close'].values
    vol = df['Volume'].values*1e-9
    k=0
    for i in range(win,len(df)):
        r = cl[i-1]/op[i-win]-1
        v = np.sum(vol[i-win:i])
        rpv[k] = r/v
        k+=1
    return rpv

def comp_rtn_per_value(df, win=10):
    rpv = np.zeros(len(df)-win)
    op = df['Open'].values
    cl = df['Close'].values
    vol = df['Volume'].values*1e-9
    val = df['Volume']*(df['High']+df['Low'])*.5 * 1e-9
    k=0
    for i in range(win,len(df)):
        r = cl[i-1]/op[i-win]-1
        v = np.sum(val[i-win:i])
        rpv[k] = r/v
        k+=1
    return rpv
def comp_latest_rpval(df,win=250):
    rpv = []
    op = df['Open'].values
    cl = df['Close'].values
    hi = df['High'].values
    lw = df['Low'].values
    vol= df['Volume'].values
    for i in range(-win,0):
        r = cl[i]/op[i] - 1
        val = (hi[i]+lw[i])*.5 * vol[i]*1e-9
        rpv.append(r/val)
    return rpv

def comp_rtn_per_value(data,lookback=10, top_ranks=20):
    df_open = data['Open']
    df_close = data['Close']
    df_high = data['High']
    df_low = data['Low']
    df_vol = data['Volume']
    df_trade_val = (df_high+df_low)*.5 * df_vol * 1e-9
    df_rpval = (df_close/df_open-1)/df_trade_val

    df_rpval = df_rpval.rolling(window=lookback,axis=0).mean()

    rpv = df_rpval.iloc[-1,:]
    sorted_ids = np.argsort(rpv)

    print("Ref UAL: ", df_rpval['UAL'].values[-1])
    print("highest 10: ", df_rpval.keys()[sorted_ids][-top_ranks:])
    print("score: ", rpv[sorted_ids][-top_ranks:])
def check_dep(data,ticker,lookback=20,lookfwd=3):
    ticker = ticker.upper()
    df_open = data['Open']
    df_close = data['Close']
    df_vol = data['Volume'] * 1e-9
    df_rpv = (df_close / df_open - 1) / df_vol
    rpv = df_rpv[ticker].rolling(window=lookback).mean().values
    labels = []
    fts = []
    # pdb.set_trace()
    prices = df_close[ticker].values
    for i in range(lookback,len(df_rpv)-lookfwd):
        fts.append(rpv[i])
        r = prices[i+lookfwd]/prices[i] - 1
        if r>0.025:
            labels.append(1)
        elif r < -0.025:
            labels.append(-1)
        else:
            labels.append(0)

    fts = np.array(fts)
    mi = mutual_info_classif(fts.reshape(-1,1),labels)
    print(mi)

def comp_rtn_per_vol(data,ticker,win=20):
    df_open = data['Open']
    df_close = data['Close']
    # df_high = data['High']
    # df_low = data['Low']
    df_vol = data['Volume']*1e-9
    df_rpv = (df_close/df_open - 1) / df_vol
    # return df_rpv

    # pdb.set_trace()
    print(f"ave rpv {ticker}: {df_rpv[ticker].mean()}")
    rpv = df_rpv[ticker].rolling(window=win).mean().values
    z = np.log(df_close[ticker].values)

    plot_double_y_axis(rpv[-500:],z[-500:])
    plt.show()

def comp_vol_weighted_rtn(ticker,lookback=10,lookfwd=5):
    back_days = 800
    target_date = datetime.today().strftime('%Y-%m-%d')
    start_date = add_days_to_date(target_date, -back_days)
    # data = yf.download(syms, start=start_date, end=target_date)
    data = yf.Ticker(ticker).history(start=start_date, end=target_date, interval='1d')
    # ticker = ticker.upper()
    df_open = data['Open'].values
    df_close = data['Close'].values
    # df_high = data['High']
    # df_low = data['Low']
    df_vol = data['Volume'] * 1e-9
    df_volsum = df_vol.rolling(window=lookback).sum().values

    vwr = []
    labels = []
    z = np.log(df_close)
    frs = []
    for i in range(lookback,len(df_vol)-lookfwd):
        r = df_close[i]/df_open[i-lookback]-1
        v = df_volsum[i]
        vwr.append(v)

        fr = z[i+lookfwd] - z[i]
        frs.append(fr)
        if fr >=0.0:
            labels.append(1)
        else:
            labels.append(0)
    vwr = np.array(vwr)
    mi=mutual_info_classif(vwr.reshape(-1,1),labels)
    print("mutual info: ", mi)
    cr = np.corrcoef(frs,vwr)[0,1]
    print("corr: ", cr)
    # return cr
    return mi[0]
    p = df_close.values[lookback:len(df_vol)-lookfwd]
    plot_double_y_axis(vwr,np.log(p))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ticker>")
        sys.exit(1)

    ticker = sys.argv[1]
    # data = pd.read_csv(DATA_FILE, comment='#', header=[0, 1], parse_dates=[0], index_col=0)
    # data = data.dropna(axis=1)

    # comp_rtn_per_value(data)
    crs = []
    for i in range(1,100):
        c = comp_vol_weighted_rtn(ticker,lookback=i)
        crs.append(c)

    plt.plot(crs,'.-')
    plt.show()
