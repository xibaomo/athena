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
rh.login(username, password)

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
    print(f"Best lookback: {best_lk}, min_diff: {mindiff:.3f}")
    return best_lk
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <target_date> [lb_rtn] [ub_rtn]")
        sys.exit(1)

    ticker = sys.argv[1]
    exp_date = sys.argv[2]
    lb_rtn = -.25
    ub_rtn = .25
    if len(sys.argv) > 4:
        lb_rtn = float(sys.argv[3])
        ub_rtn = float(sys.argv[4])

    df_option = get_option_chain(ticker,exp_date)
    cur_price = float(rh.stocks.get_latest_price(ticker)[0])
    print(f"Latest price: {cur_price:.2f}")

    df, bars_per_day = download_from_yfinance(ticker, period='2y')
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    trade_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {trade_days}")
    steps = trade_days*bars_per_day

    lookback_days = findBestLookbackDays(22,22*18,trade_days,bars_per_day,rtns)
    rtns = rtns[-lookback_days*bars_per_day:]

    calibrateStrikePrice(df_option,steps,rtns,cur_price)

