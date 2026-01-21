import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys, os
from cal_prob import findBestLookbackDays, prepare_rtns
from utils import *
from mkv_cal import *
import matplotlib.pyplot as plt
from option_chain import *
import random
from covered_call import sliding_cdf_error, calibrate_weights


def compute_revenue(price, s, d):
    '''
    compute revenue for given price, strike and d
    '''

    if price < s + d and price > s:
        return s + d - price
    if price <= s and price > s - d:
        return price - (s - d)

    return 0


def compute_cost(s, d, f_ask, f_bid):
    income = 2 * f_bid(s)  # sell 2 at s
    cost = f_ask(s + d) + f_ask(s - d)  # buy 2 options
    return cost - income


def compute_expected_profit(s, d, ask_cal, bid_cal, probs, lb_rtn, ub_rtn, p0):
    cost = compute_cost(s, d, ask_cal, bid_cal)
    drtn = (ub_rtn - lb_rtn) / len(probs)
    exp_rtn = 0.
    for i in range(len(probs)):
        rtn = (i + .5) * drtn + lb_rtn
        price = p0 * (1. + rtn)
        revenue = compute_revenue(price, s, d)
        rtn = (revenue - cost) / (s * 2 + cost)
        exp_rtn += rtn * probs[i]

    return exp_rtn


def calibrate_butterfly(p0, lb_rtn, ub_rtn, probs, ask_cal, bid_cal, strike_bounds):
    def obj_func(x, params):
        s, d = x
        _p0, _ask_cal, _bid_cal, _lb_rtn, _ub_rtn, _probs = params
        exp_rtn = compute_expected_profit(s, d, _ask_cal, _bid_cal, _probs, _lb_rtn, _ub_rtn, _p0)

        return -exp_rtn

    up_d = 100  ##distance of top put from sold put
    print(f"max d: {up_d:.2f}")
    strike_bounds[0] = (strike_bounds[0] + p0) * .5
    if strike_bounds is not None:
        bounds = [strike_bounds, (10, up_d)]
    max_rtn = -999
    opt_x = [0, 0]
    for i in range(10):
        x0 = np.array([p0, random.random() * up_d])
        res = minimize(obj_func, x0, args=((p0, ask_cal, bid_cal, lb_rtn, ub_rtn, probs),), bounds=bounds)
        print(f"optimal s,d: {res.x}")
        print(f"maximized expected return: {-res.fun:.2f}")

        if -res.fun > max_rtn and -res.fun < 10:
            max_rtn = -res.fun
            opt_x = res.x

    return opt_x, max_rtn


def prepare_puts(sym, exp_date):
    options = get_option_chain_alpha_vantage(sym)
    print(f"{len(options)} options downloaded")
    puts = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'put':
            puts.append(opt)

    print(f"{len(puts)} calls returned")
    # for call in calls:
    #     plt.plot(float(call['strike']),float(call['bid']),'.')
    # plt.show()
    return puts


def create_premium_cal(options, p0, type):
    strikes = []
    asks = []
    bids = []
    for optn in options:
        strikes.append(float(optn['strike']))
        asks.append(float(optn['ask']))
        bids.append(float(optn['bid']))
    strikes = np.array(strikes)
    asks = np.array(asks)
    bids = np.array(bids)
    f_ask = interp1d(strikes, asks, kind='cubic', fill_value="extrapolate")
    f_bid = interp1d(strikes, bids, kind='cubic', fill_value="extrapolate")
    bounds = [np.min(strikes), np.max(strikes)]
    # ask_cal = lambda x: f_ask(x) if x <= bounds[1] else  99990.
    # ask_cal = lambda x: 0 if x <= bounds[0] elif x>=bounds[1] x-p0 else f_ask(x)
    if type == "put":
        ask_cal = lambda x: 0 if x <= bounds[0] else (x - p0 if x >= bounds[1] else f_ask(x))
        bid_cal = lambda x: 0 if x <= bounds[0] else (x - p0 if x >= bounds[1] else f_bid(x))
    if type == "call":
        ask_cal = lambda x: 0 if x >= bounds[1] else (p0 - x if x <= bounds[0] else f_ask(x))
        bid_cal = lambda x: 0 if x >= bounds[1] else (p0 - x if x <= bounds[0] else f_bid(x))

    return ask_cal, bid_cal, bounds

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <expiration_date>")
        sys.exit(0)

    ticker = sys.argv[1]
    exp_date = sys.argv[2]
    lb_rtn = -0.5
    ub_rtn = 1.

    fwd_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {fwd_days}")
    df, bars_per_day = download_from_yfinance(ticker)

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    breakpoint()
    p0 = df['Close'].values[-1][0]
    print(f"Latest price: {p0}")
    calls,puts = prepare_callsputs(ticker,exp_date)
    ask_cal, bid_cal, strike_bounds = create_premium_cal(puts, p0,'put')

    lookback_days, min_diff = findBestLookbackDays(22 * 12, 22 * 22, fwd_days, bars_per_day, rtns)
    print(f"optimal days: {lookback_days}, min_diff: {min_diff}")

    picked_rtns = rtns[-lookback_days * bars_per_day:]

    # probs = compute_steady_dist(picked_rtns,lb_rtn=-0.5,ub_rtn=1.)
    cdfcal = ECDFCal(picked_rtns)
    probs = compMultiStepProb(fwd_days * bars_per_day, lb_rtn=lb_rtn, ub_rtn=ub_rtn, cdf_cal=cdfcal)
    plt.plot(probs, '.')

    # breakpoint()
    cdfcal = compute_historical_distribution(rtns,fwd_days, bars_per_day)
    probs = compMultiStepProb(fwd_days * bars_per_day, lb_rtn=lb_rtn, ub_rtn=ub_rtn, cdf_cal=cdfcal)

    res, exp_rtn = calibrate_butterfly(p0, lb_rtn=lb_rtn, ub_rtn=ub_rtn, probs=probs, ask_cal=ask_cal, bid_cal=bid_cal,
                                       strike_bounds=strike_bounds)

    s, d = res
    print(f"optimal s: {s:.2f},d: {d:.2f}, s-d: {s - d:.2f}, s+d: {s + d:2f}")
    cost = compute_cost(s, d, ask_cal, bid_cal)
    print(f"position cost: {cost:.2f}")
    print(f"expected return: {exp_rtn:.4f}")
    print(
        f"max daily return: {exp_rtn / fwd_days:.4f}, annual return: {exp_rtn / fwd_days * 252:.4f}")
    print(f"sym: {ticker}, latest price: {p0:.2f}")
    plt.show()
