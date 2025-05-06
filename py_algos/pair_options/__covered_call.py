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
from cal_prob import prepare_rtns,compute_latest_dist_diff,findBestLookbackDays
import matplotlib.pyplot as plt
import yfinance as yf

def __get_option_chain(ticker, expiration_date, type='call'):
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
def get_option_chain(ticker_sym, expiration_date, type='call'):
    ticker = yf.Ticker(ticker_sym)
    if not expiration_date in ticker.options:
        print("expiration date not found: ", expiration_date)
        print("Available date: ", ticker.options)
        sys.exit(1)
    opchain = ticker.option_chain(expiration_date)
    df = opchain.puts
    if type == 'call':
        df = opchain.calls

    return df,ticker

def compExpectedReturn(probs, strike_rtn,bid_rtn, lb_rtn,ub_rtn,nstates=500):
    d = (ub_rtn-lb_rtn)/(nstates-1)
    exp_rtn = 0.
    for i in range(nstates):
        r = lb_rtn+i*d+ 1
        if r > strike_rtn:
            r = strike_rtn
        exp_rtn += (r+bid_rtn)*probs[i]
    return exp_rtn
def calibrateStrikePrice(df_options, steps, rtns, cur_price, nstates=500):
    strike_key = 'strike'
    bid_key = 'lastPrice'
    df = df_options.sort_values(by=strike_key)
    strikes = df[strike_key].values
    print(f"max strike: {strikes[-1]}")
    lb_rtn = strikes[0] / cur_price - 1.
    ub_rtn = strikes[-1] / cur_price - 1.
    cal = MkvRegularCal(nstates=nstates)
    probs = cal.compMultiStepProb(steps=steps, rtns=rtns, lb_rtn=lb_rtn, ub_rtn=ub_rtn)
    print(f"sum of probs: {np.sum(probs)}")
    # x = np.linspace(-.25,.25,500)
    # plt.plot(x,probs,'.')
    # plt.show()
    best_rtn = -9999
    best_strike = -1
    for i in range(len(df)):
        # pdb.set_trace()
        strike = strikes[i]
        # if strike < cur_price:
        #     continue
        strike_rtn = strike/cur_price
        bid = df[bid_key].values[i]
        # if bid == 0:
        #     continue
        bid_rtn = df[bid_key].values[i]/cur_price
        exp_rtn = compExpectedReturn(probs,strike_rtn,bid_rtn, lb_rtn, ub_rtn, nstates)
        # print(f"{strike},  {bid}, {exp_rtn-1}")
        if exp_rtn > best_rtn:
            best_rtn = exp_rtn
            best_strike = strike
            best_bid = bid
    print(f"Best strike: {best_strike}, best bid: {best_bid}, highest expected rtn: {best_rtn-1}")
    return best_rtn-1

# def findBestLookbackDays(lb_days, ub_days, fwd_days,bars_per_day, rtns, intvl=5):
#     xs = np.arange(lb_days, ub_days, intvl)
#     ys = []
#     mindiff = 99999
#     best_lk = -1
#     for x in xs:
#         y = compute_latest_dist_diff(x, fwd_days, bars_per_day, rtns)
#         ys.append(y)
#         if y < mindiff:
#             mindiff = y
#             best_lk = x
#     print(f"Best lookback: {best_lk}, min_diff: {mindiff:.3f}")
#     return best_lk
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <target_date> [lb_rtn] [ub_rtn]")
        sys.exit(1)

    ticker_str = sys.argv[1]
    exp_date = sys.argv[2]
    lb_rtn = -.25
    ub_rtn = .25
    if len(sys.argv) > 4:
        lb_rtn = float(sys.argv[3])
        ub_rtn = float(sys.argv[4])

    df_option,ticker = get_option_chain(ticker_str,exp_date)
    # cur_price = float(rh.stocks.get_latest_price(ticker)[0])
    cur_price = ticker.history(period="1d")["Close"].iloc[-1]
    print(f"Latest price: {cur_price:.2f}")

    df, bars_per_day = download_from_yfinance(ticker_str, period='2y')
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    print("bars per day: ", bars_per_day)
    trade_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {trade_days}")
    steps = trade_days*bars_per_day

    lookback_days = findBestLookbackDays(22,22*18,trade_days,bars_per_day,rtns)
    rtns = rtns[-lookback_days*bars_per_day:]

    best_rtn = calibrateStrikePrice(df_option,steps,rtns,cur_price)
    print(f"expected daily rtn: {best_rtn/trade_days}")

