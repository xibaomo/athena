import os, sys
import pdb
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf
from scipy import stats
from cal_prob import findBestLookbackDays, prepare_rtns
from utils import *
from mkv_cal import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from option_chain import *
from covered_call import sliding_cdf_error, calibrate_weights, evaluate_latest_wasserstein_distance


def compExpectedReturn(cur_price, sell_strike, buy_strike, ask,bid, probs, drtn, lb_rtn, n_sell):
    expected_rtn = 0.
    cost = cur_price * n_sell + ask #cost is stock cost for covered call

    for i in range(len(probs)):
        r = lb_rtn + (i + 0.5) * drtn
        p = cur_price * (1 + r)
        prf = 0.
        if p > sell_strike:
            prf = (bid -  (p - sell_strike))*n_sell + (p - buy_strike - ask)
        elif p < buy_strike:
            prf = 0. + bid*n_sell
        else:
            # prf = (p - buy_strike - ask) + bid*n_sell
            prf = bid*n_sell
        # if sell_strike == 250 and rev < 0:
        #     breakpoint()
        rtn = prf/cost
        expected_rtn = expected_rtn + probs[i] * rtn
    return expected_rtn

def calibrate_call_strike(cdf_cal, cur_price, calls, steps, lb_rtn, ub_rtn, ratio=2 ):
    max_rtn = -999990.0
    best_idx = []

    probs = compMultiStepProb(steps, lb_rtn, ub_rtn, cdf_cal)
    x = np.linspace(lb_rtn, ub_rtn, len(probs))
    plt.plot(x, probs, '.')

    drtn = (ub_rtn - lb_rtn) / len(probs)
    for i in range(len(calls)-1,-1,-1):
        call = calls[i]
        sell_strike = float(call['strike'])
        bid = float(call['bid'])

        for j in range(len(calls)):
            ask =  float(calls[j]['ask'])
            # premium = 0
            buy_strike = float(calls[j]['strike'])

            exp_rtn = compExpectedReturn(cur_price, sell_strike=sell_strike, buy_strike=buy_strike,
                                         ask=ask,bid=bid, probs=probs, drtn=drtn,
                                         lb_rtn=lb_rtn, n_sell=ratio)

            # print(
            #     f"strike pair: {sell_strike},{buy_strike}, exp_rtn: {exp_rtn:.4f}, prem: {call['bid']},{calls[j]['ask']}")
            if exp_rtn > max_rtn:
                max_rtn = exp_rtn
                best_idx = [i,j]

    return best_idx, max_rtn

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <expiration_date>")
        sys.exit(0)

    ticker = sys.argv[1]
    exp_date = sys.argv[2]

    fwd_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {fwd_days}")
    df, bars_per_day = download_from_yfinance(ticker, period='730d', interval='1h')

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    print(f"length of rtns: {len(rtns)}, bars_per_day: {bars_per_day}")
    cur_price = df['Close'].values[-1][0]

    # spacing,min_diff = find_stablest_spacing(rtns,22*bars_per_day,2*bars_per_day)
    # print(f"length of rtns: {len(rtns)}, min ave diff: {min_diff}, spacing days: {spacing//bars_per_day}")

    lookback_days, min_diff = findBestLookbackDays(22 * 6, 730, fwd_days, bars_per_day, rtns)
    print(f"optimal days: {lookback_days}, min_diff: {min_diff}")

    pick_rtns = rtns[-lookback_days * bars_per_day:]

    calls, puts = prepare_callsputs(ticker, exp_date)
    call_put_ratio = call_put_ask_ratio(0.25, calls, puts)
    print(f"0.25_delta P/C ratio: {1./call_put_ratio:.3f}")
    # pdb.set_trace()

    steps = fwd_days * bars_per_day
    cdf_cal = ECDFCal(pick_rtns)
    ratio = 2.
    best_idx, max_rtn = calibrate_call_strike(cdf_cal, cur_price, calls, steps,
                                                            lb_rtn=-0.5, ub_rtn=1., ratio=ratio )
    print(f"Latest price: {cur_price:.2f}")
    sell_call = calls[best_idx[0]]
    buy_call = calls[best_idx[1]]
    print(f"best strike: {sell_call['strike']},{buy_call['strike']} , max_rtn: {max_rtn}")
    print(f"bid: {sell_call['bid']}, ask: {buy_call['ask']}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")

    cdf_cal = compute_historical_distribution(rtns,fwd_days,bars_per_day)
    best_idx, max_rtn = calibrate_call_strike(cdf_cal, cur_price, calls, steps,
                                                            lb_rtn=-0.5, ub_rtn=1., ratio=ratio )
    print(f"Latest price: {cur_price:.2f}")
    sell_call = calls[best_idx[0]]
    buy_call = calls[best_idx[1]]
    print(f"best strike: {sell_call['strike']},{buy_call['strike']} , max_rtn: {max_rtn}")
    print(f"bid: {sell_call['bid']}, ask: {buy_call['ask']}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")

    print(f"sym: {ticker}, latest price: {cur_price:.2f}")
    plt.show()
