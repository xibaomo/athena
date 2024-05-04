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
import numpy as np
from scipy.signal import savgol_filter
import pywt
import math
from scipy.stats import spearmanr
from scipy.stats import shapiro

def wavelet_smoothing(data, wavelet='db4', level=2):
    """Smooth a curve using wavelet transform."""
    # Perform discrete wavelet transform
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Threshold the wavelet coefficients
    threshold = np.std(coeffs[-1]) / 2
    new_coeffs = [coeff if isinstance(coeff, np.ndarray) else pywt.threshold(coeff, threshold, mode='soft') for coeff in
                  coeffs]

    # Reconstruct the smoothed signal
    smoothed_signal = pywt.waverec(new_coeffs, wavelet)

    return smoothed_signal

def savitzky_golay_smoothing(data, window_size, order):
    """Smooth a curve using the Savitzky-Golay filter."""
    smoothed_data = savgol_filter(data, window_size, order)
    return smoothed_data

NUM_SYMS=50
from scipy.stats import norm

def gaussian_kernel(x, x_data, y_data, h):
    """Calculate the Gaussian kernel function."""
    kernel_values = norm.pdf((x - x_data) / h)
    normalized_kernel = kernel_values / np.sum(kernel_values)
    smoothed_value = np.sum(normalized_kernel * y_data)
    return smoothed_value

def gaussian_kernel_smoothing(x_data, y_data, h):
    """Smooth the data using Gaussian kernel smoothing."""
    smoothed_data = []
    for x in x_data:
        smoothed_value = gaussian_kernel(x, x_data, y_data, h)
        smoothed_data.append(smoothed_value)
    return smoothed_data
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
    # pdb.set_trace()
    smdata = y1.copy()
    x = np.arange(len(y1))
    smdata[:] = savitzky_golay_smoothing(y1.values,31,2)
    # smdata[:] = wavelet_smoothing(y1.values,wavelet="db4",level=5)[:len(smdata)]

    ax1.plot(smdata,'y',linewidth=2)

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

def corr_lookback(lookback,df_sh,df_close):
    x = df_sh.rolling(window=int(lookback)).mean().values
    # pdb.set_trace()
    idx = ~np.isnan(x)
    x = x[idx]
    y = df_close.values[idx]
    # cr =  np.corrcoef(x,y)[0,1]
    cr,pv = spearmanr(x,y)
    print("curretn lookback: ", lookback,cr)
    return -cr
def optimize_lookback(df_sh,df_close):
    b = 500
    a = 10
    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
    (a, b) = (min(a, b), max(a, b))
    h = b - a

    c = a + invphi2 * h
    d = a + invphi * h
    yc = corr_lookback(c,df_sh,df_close)
    yd = corr_lookback(d,df_sh,df_close)

    while b-a>1:
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = corr_lookback(c,df_sh,df_close)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = corr_lookback(d,df_sh,df_close)

    print("best lookback: ",b,-corr_lookback(int(b),df_sh,df_close))
    return int(b)


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
    df_volval = df_typ*df_vol
    daily_rtn = df_typ.pct_change()
    # zero_cols = df_vol.columns[(df_vol==0).any()]
    # df_vol = df_vol.drop(zero_cols,axis=1)
    # daily_rtn = daily_rtn.drop(zero_cols,axis=1)

    df_sh0 = daily_rtn/df_volval
    # df_sh = daily_rtn

    lookback = optimize_lookback(df_sh0, df_close)
    # lookback = 100
    df_sh = df_sh0.rolling(window=lookback).mean()
    x=df_sh.iloc[:].values
    idx = ~np.isnan(x)
    x = x[idx]
    y = df_close.values[idx]
    # print(adfuller(x))

    print("corr: ",np.corrcoef(x,y))
    print("mi: ",mutual_info_regression(x[:-1].reshape([-1,1]),y[1:]))
    plot_double_y_axis(df_sh.iloc[:],df_close.iloc[:])
    # print("sm corr: ", np.corrcoef(smdata,y))



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




