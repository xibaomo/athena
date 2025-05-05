import pdb
import pickle,sys,os
import time
import random
from datetime import datetime,timedelta
from scipy.optimize import minimize
import numpy as np
from mkv_cal import MkvRegularCal
import pandas as pd
from utils import TradeDaysCounter,download_from_yfinance
from cal_prob import prepare_rtns,compute_latest_dist_diff
import matplotlib.pyplot as plt

import robin_stocks.robinhood as rh
# Login to Robinhood
username = os.getenv("BROKER_USERNAME")
password = os.getenv("BROKER_PASSWD")
mfa_code="131085"
# login=rh.login(username, password, store_session=True)
login=rh.login(username, password, mfa_code=mfa_code)
print(login)

def get_option_chain(ticker, expiration_date, type='call'):
    """
    Fetches the options chain for a given ticker and expiration date.
    """
    # Fetch the options for the given ticker and expiration date
    options = rh.options.find_options_by_expiration(ticker, expiration_date)

    # Convert the data to a Pandas DataFrame for analysis
    df = pd.DataFrame(options)
    if type=='call':
        df = df.drop(df[df['type'] == 'put'].index)
    if type == 'put':
        df = df.drop(df[df['type'] == 'call'].index)
    df['strike_price'] = df['strike_price'].astype(float)
    df['bid_price']  = df['bid_price'].astype(float)
    df['ask_price'] = df['ask_price'].astype(float)

    df = df.sort_values(by='strike_price')
    return df

def compExpectedReturn(probs, strike_rtn,bid_rtn, lb_rtn,ub_rtn,nstates=500):
    d = (ub_rtn-lb_rtn)/nstates
    exp_rtn = 0.
    for i in range(nstates):
        r = lb_rtn+i*d+.5*d + 1
        if r > strike_rtn:
            r = strike_rtn
        exp_rtn += (r+bid_rtn)*probs[i]
    return exp_rtn
def calibrateStrikePrice(df_options, steps, rtns, cur_price, lb_rtn=-.25, ub_rtn=.25, nstates=500):
    sorted_df = df_options.sort_values(by='strike_price')
    cal = MkvRegularCal(nstates=nstates)
    probs = cal.compMultiStepProb(steps=steps, rtns=rtns, lb_rtn=lb_rtn, ub_rtn=ub_rtn)
    print(f"sum of probs: {np.sum(probs)}")
    # x = np.linspace(-.25,.25,500)
    # plt.plot(x,probs,'.')
    # plt.show()
    best_rtn = -9999
    best_strike = -1
    for i in range(len(sorted_df)):
        # pdb.set_trace()
        strike = sorted_df['strike_price'].values[i]
        if strike < cur_price:
            continue
        strike_rtn = strike/cur_price
        bid = sorted_df['bid_price'].values[i]
        # if bid == 0:
        #     continue
        bid_rtn = sorted_df['bid_price'].values[i]/cur_price
        exp_rtn = compExpectedReturn(probs,strike_rtn,bid_rtn, lb_rtn, ub_rtn, nstates)
        print(f"{strike}, , {bid}, {exp_rtn}")
        if exp_rtn > best_rtn:
            best_rtn = exp_rtn
            best_strike = strike
    print(f"Best strike: {best_strike}, highest rtn: {best_rtn}")

def findBestLookbackDays(lb_days, ub_days, fwd_days,bars_per_day, rtns, intvl=5):
    xs = np.arange(lb_days, ub_days, intvl)
    ys = []
    mindiff = 99999
    best_lk = -1
    for x in xs:
        y = compute_latest_dist_diff(x, fwd_days, bars_per_day, rtns)
        ys.append(y)
        if y < mindiff:
            mindiff = y
            best_lk = x
    print(f"Best lookback days: {best_lk}, min_diff: {mindiff:.3f}")
    return best_lk

def compExpRevenue(probs, cur_price, strike, gap,drtn,lb_rtn):
    lw_rtn = (strike-gap)/cur_price - 1.
    hi_rtn = (strike+gap)/cur_price - 1.
    # mid_rtn = strike/cur_price -1.
    lw = int((lw_rtn-lb_rtn)/drtn)
    hi = int((hi_rtn-lb_rtn)/drtn)

    # pdb.set_trace()
    s = 0
    for i in range(lw,hi+1):
        price = (lb_rtn + i*drtn + 1.)*cur_price
        rev = 0
        if price < strike:
            rev = price - strike + gap
        else:
            rev = strike + gap - price
        s+= rev*probs[i]
    return s

def findOptimalButterfly(df,cur_price,rtns,steps,ns=500):
    df = df.sort_values(by='strike_price')
    cal = MkvRegularCal(nstates=ns)
    ask = df['ask_price'].values
    strike = df['strike_price'].values
    lb_rtn = strike[0]/cur_price - 1.
    ub_rtn = strike[-1]/cur_price - 1.
    probs = cal.compMultiStepProb(steps, rtns,lb_rtn,ub_rtn)
    drtn = (ub_rtn-lb_rtn)/(ns-1)
    print(f"rtn interval: {drtn}")

    # x = np.linspace(lb_rtn,ub_rtn,ns)
    # plt.plot(x,probs,'.-')
    # plt.show()
    print(f"sum of probs: {np.sum(probs)}")
    max_rtn = -99999999
    center_strike = -1
    bestgap = 0
    for i in range(1,len(df)):
        pd = i-1
        while pd >= 0:
            pu = i + 1
            while pu < len(df):
                if strike[i]*2 == strike[pu] + strike[pd]:
                    break
                pu += 1
            if pu == len(df):
                pd -= 1
                continue
            #compute expected revenue
            gap = strike[i] - strike[pd]
            cost = ask[pu] + ask[pd] - df['bid_price'].values[i] * 2
            exp_rev = compExpRevenue(probs, cur_price, strike[i], gap, drtn, lb_rtn)
            profit = exp_rev - cost
            r = profit/cost

            if r > max_rtn and cost > 1 and cost < 5:
            # if r > max_rtn:
                max_rtn = r
                center_strike = strike[i]
                bestgap = gap
                bestcost = cost
            pd -= 1

    print(f"max exp rtn: {max_rtn}, cost: {bestcost}")
    print(f"center strike: {center_strike}, gap: {bestgap}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <target_date> ")
        sys.exit(1)

    ticker = sys.argv[1]
    exp_date = sys.argv[2]


    df = get_option_chain(ticker,exp_date)
    cur_price = float(rh.stocks.get_latest_price(ticker)[0])
    print(f"Latest price: {cur_price:.2f}")

    dfp, bars_per_day = download_from_yfinance(ticker, period='2y')
    rtns, bars_per_day = prepare_rtns(dfp, bars_per_day)
    trade_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {trade_days}")
    steps = trade_days*bars_per_day

    lookback_days = findBestLookbackDays(22,22*18,trade_days,bars_per_day,rtns)
    rtns = rtns[-lookback_days*bars_per_day:]

    findOptimalButterfly(df,cur_price,rtns,steps)

