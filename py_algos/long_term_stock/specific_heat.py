import pandas as pd
import os,sys
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.fundamentaldata import FundamentalData
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller, kpss
import pdb
API_KEY='ZHXCW0CK7QJFMHK0'
NUM_SYMS=50

def plot_double_y_axis(y1, y2, xlabel='X', y1label='Y1', y2label='Y2', title='Double Y-Axis Plot'):
    fig, ax1 = plt.subplots()

    # Plot the first dataset with primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=color)
    ax1.plot(y1,'.-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    mu = np.mean(y1)
    sd = np.std(y1)
    ax1.axhline(y=mu,linestyle='--')
    ax1.axhline(y=mu+2*sd,linestyle='--')
    ax1.axhline(y=mu-2*sd,linestyle='--')

    # Create a secondary y-axis and plot the second dataset
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel(y2label, color=color)
    ax2.plot(y2, '.-',color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    plt.show()
def add_days_to_date(date_str, num_days):
    # Convert string to datetime object
    # pdb.set_trace()
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the resulting date back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str
def cal_typical_price(data):
    df_high = data['High']
    df_low = data['Low']
    df_close = data['Close']
    df_typ = (df_high+df_low+df_close)/3
    return df_typ
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} <date> <lookback_days> <sym> ")
        sys.exit(1)

    target_date = sys.argv[1]

    # df = pd.read_csv('sp505.csv', comment='#')
    # syms = df['<SYM>'].values

    # syms = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    # syms.append('TSM')
    syms=[]

    lookback = int(sys.argv[2])
    syms.append(sys.argv[3])
    start_date = add_days_to_date(target_date, -lookback)
    end_date = target_date
    print("history range: ", start_date, end_date)
    data = yf.download(syms, start=start_date, end=end_date)
    data = data.dropna(axis=1)
    df_close = data['Close']

    df_typ = cal_typical_price(data)
    df_vol = data['Volume']
    daily_rtn = df_typ.pct_change()
    # zero_cols = df_vol.columns[(df_vol==0).any()]
    # df_vol = df_vol.drop(zero_cols,axis=1)
    # daily_rtn = daily_rtn.drop(zero_cols,axis=1)

    df_sh = daily_rtn/df_vol
    # df_sh = daily_rtn   # pdb.set_trace()
    lookback=20
    df_sh = df_sh.rolling(window=lookback).mean()
    x=df_sh.iloc[:].values
    idx = ~np.isnan(x)
    x = x[idx][:-1]
    y = df_close.values[idx][1:]
    print(adfuller(x))
    print("corr: ",np.corrcoef(x,y))
    print("mi: ",mutual_info_regression(x.reshape([-1,1]),y))
    plot_double_y_axis(df_sh.iloc[:],df_close.iloc[:])



    sys.exit(0)
    xx = [x for x in range(20)]
    for sym in df_sh.keys():
        val = df_sh[sym].values
        slopes = []
        for i in range(lookback+20,len(val)-20):
            coef = np.polyfit(xx,val[i-20:i],1)[0]
            slopes.append(coef)
        y = df_close[sym].pct_change()[lookback+20:]
        sh = val[lookback+20:-20]
        mu = np.zeros(len(slopes))
        for i in range(len(slopes)):
            mu[i] = sum(y[i:i+20])
        lbs = [1 if yy > 0 else 0 for yy in y]
        lbs = np.array(lbs)
        slopes = np.array(slopes)
        # mask = ~np.isnan(slopes)
        # slopes =slopes[mask]
        # y = y[mask]
        # lbs = lbs[mask]
        cm = np.corrcoef(slopes,mu)[0,1]
        mi = mutual_info_regression(slopes.reshape([-1,1]),mu)
        print(sym,cm,mi)
        cm = np.corrcoef(sh, mu)[0, 1]
        mi = mutual_info_regression(sh.reshape([-1, 1]), mu)
        print(sym, cm, mi)
        if mi > 0.5:
            plot_double_y_axis(sh,df_close[sym].values[lookback+20:-20])




