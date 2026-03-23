import numpy as np
import pandas as pd
import statsmodels.sandbox.distributions.gof_new

from mkv_cal import *
from option_chain import *
from utils import *
from arch import arch_model
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def prepare_rtns(df, tid, lookback, ticker):
    if tid == -1:
        tid = len(df)-1
    k = 0
    pc = np.zeros(lookback * 3)
    for i in range(tid - lookback, tid):
        pc[k] = df['Open'][ticker][i]
        k = k + 1
        pc[k] = (df['High'][ticker][i] + df['Low'][ticker][i]) / 2
        k = k + 1
        pc[k] = df['Close'][ticker][i]
        k = k + 1
    rtns = np.zeros(lookback* 3 )
    for i in range(1, len(pc)):
        rtns[i - 1] = (pc[i] - pc[i - 1]) / pc[i - 1]
    # breakpoint()
    return rtns


def compute_rtn_each_trade(tid, ticker, df, lookback, horizon, asgn_thd, calls, lb_rtn=-0.5, ub_rtn=1.):
    # breakpoint()
    rtns = prepare_rtns(df, tid, lookback,ticker)
    cdf_cal = ECDFCal(rtns)
    probs = compMultiStepProb(horizon, lb_rtn, ub_rtn, cdf_cal)

    drtn = (ub_rtn - lb_rtn) / len(probs)
    picked_strike_rtn = 0
    picked_bid = 0
    for call in calls:
        s_rtn = call['rtn']
        if s_rtn < lb_rtn:
            continue
        if s_rtn > ub_rtn:
            breakpoint()
        idx = (s_rtn - lb_rtn) // drtn
        idx = int(idx)
        if idx < 0:
            breakpoint()
        pb_asgn = sum(probs[idx:])
        if pb_asgn <= asgn_thd:
            picked_strike_rtn = s_rtn
            picked_bid = float(call['bid'])
            break

    # breakpoint()
    pos_price = df.iloc[tid]['Close'][ticker]
    strike = pos_price * (1 + picked_strike_rtn)

    exp_price = df.iloc[tid + horizon]['Close'][ticker]

    if exp_price > strike:
        profit = strike - exp_price + picked_bid
    else:
        profit = picked_bid
    print(f"{df.index[tid]}, {profit:.2f}")
    return profit / pos_price


def back_test(ticker, df, fwd_days, bars_per_day, lookback_days, calls, year=2025):
    lookback = lookback_days * bars_per_day
    horizon = fwd_days * bars_per_day
    # breakpoint()

    # 1. Get all entries that fall on a Monday
    mondays_all = df[(df.index.weekday == 0) & (df.index.year == year)]

    # 2. Group by date and take the first row.
    # .apply(lambda x: x.index[0]) extracts the ACTUAL timestamp label from the original DF.
    monday_timestamps = mondays_all.groupby(mondays_all.index.date).apply(lambda x: x.index[0])

    # 3. Now get_loc will work because the labels are identical
    tids = [df.index.get_loc(ts) for ts in monday_timestamps]
    rtns = []
    for tid in tids:
        # breakpoint()
        r = compute_rtn_each_trade(tid, ticker, df, lookback, horizon, 0.4, calls)
        rtns.append(r)

    print(f"average rtn: {np.mean(rtns)}")
    return rtns


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <expiration_date>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    exp_date = sys.argv[2]
    cur_price = float(rh.stocks.get_latest_price(ticker)[0])
    print(f"Latest price: {cur_price:.2f}")

    fwd_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {fwd_days}")

    df, bars_per_day = download_from_yfinance(ticker)

    calls, puts = prepare_callsputs(ticker, exp_date)
    for call in calls:
        call['rtn'] = float(call['strike']) / cur_price - 1.

    back_test(ticker, df, fwd_days, bars_per_day, lookback_days=300, calls=calls)






