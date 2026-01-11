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


def compExpectedReturn_call_spread(cur_price, s_strike, b_strike, bid,ask, probs, drtn, lb_rtn):
    expected_rtn = 0.
    cost = ask
    # if s_strike==490 and b_strike == 490:
    #     breakpoint()

    for i in range(len(probs)):
        r = lb_rtn + (i + 0.5) * drtn
        p = cur_price * (1 + r)
        if p > s_strike:
            income_sell = bid
            cost+= p-s_strike
        else:
            income_sell = bid
        if p > b_strike:
            income_buy = p - b_strike
        else:
            income_buy = 0
        income = income_buy + income_sell
        rtn = income/cost - 1.
        expected_rtn = expected_rtn + rtn*probs[i]
    return expected_rtn

def calibrate_call_strikes(cur_price, options, steps, lb_rtn, ub_rtn, cdf_cal):
    max_rtn = -999990.0
    best_strike = []
    best_askbid = []

    probs = compMultiStepProb(steps, lb_rtn, ub_rtn, cdf_cal)

    x = np.linspace(lb_rtn, ub_rtn, len(probs))
    plt.plot(x, probs, '.')
    # plt.show()
    # pdb.set_trace()
    drtn = (ub_rtn - lb_rtn) / len(probs)
    for i in range(len(options)):
        optn_sell = options[i]
        sell_strike = optn_sell['strike']
        bid = optn_sell['bid']
        if bid < 1.:
            continue
        for j in range(len(options)):
            optn_buy = options[j]
            buy_strike = float(optn_buy['strike'])
            # if buy_strike > cur_price:
            #     continue
            if buy_strike > sell_strike:
                continue
            ask = float(optn_buy['ask'])

            exp_rtn = compExpectedReturn_call_spread(cur_price, s_strike=sell_strike, b_strike=buy_strike, bid=bid, ask=ask,
                                                     probs=probs, drtn=drtn,lb_rtn=lb_rtn)

            print(f"buy_call {buy_strike},sell_call {sell_strike}, exp_rtn: {exp_rtn:.4f}")
            if exp_rtn > max_rtn:
                max_rtn = exp_rtn
                best_strike = [buy_strike, sell_strike]
                best_askbid = [ask,bid]

    return best_strike, max_rtn, best_askbid


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
    cur_price = df['Close'].values[-1][0]

    calls, puts = prepare_callsputs(ticker, exp_date)
    call_put_ratio = call_put_ask_ratio(0.25, calls, puts)
    print(f"0.25_delta P/C ratio: {1./call_put_ratio:.3f}")
    # pdb.set_trace()

    steps = fwd_days * bars_per_day
    print(f"searching for best lookback days...")
    m_range = range(fwd_days * bars_per_day, 22 * 5 * bars_per_day)
    res = find_best_m_given_n(rtns, fwd_days * bars_per_day, m_range)
    print(f"best lookback days: {res['m'] / bars_per_day:.2f}, max corr: {res['corr']:.4f}")
    # breakpoint()

    backdays = round(res['m'] / bars_per_day)
    n_back = res['m']
    horizon = fwd_days * bars_per_day

    dmin = 99999.
    best_n_back = 0
    for i in [1, 2, 3]:
        d = evaluate_latest_wasserstein_distance(rtns, n_back * i, horizon)
        print(f"lookback: {n_back * i}, wasserstein distance: {d:.5f}")
        if d < dmin:
            dmin = d
            best_n_back = i * n_back

    n_back = best_n_back
    print(f"Searching for subarray ({n_back / bars_per_day} days) with the most likely distribution...")

    x = rtns[-n_back:]
    y = rtns[:-n_back]

    res = analog_distribution_forecast(x, y, horizon, K=5)
    pick_rtns = res['future_samples']

    cdf_cal = ECDFCal(pick_rtns)
    best_strike, max_rtn, best_askbid = calibrate_call_strikes(cur_price,calls, steps,
                                                            lb_rtn=-0.5, ub_rtn=1.,cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, ask&bid: {best_askbid}, max_rtn: {max_rtn}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")

    plt.show()
