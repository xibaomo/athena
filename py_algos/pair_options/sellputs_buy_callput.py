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


def compute_revenue(price, sc, sp, s0, s0_ratio):
    '''
    compute revenue for given price
    '''

    rev_s0 = 0 if price > s0 else (price - s0) * s0_ratio

    rev_call = 0 if price < sc else price - sc
    rev_put = 0 if price > sp else sp - price

    return rev_call + rev_put + rev_s0


def compute_cost(sc, sp, s0, call_ask, put_ask, put_bid, s0_ratio):
    cost = call_ask(sc) + put_ask(sp) - put_bid(s0) * s0_ratio
    return cost


def compute_expected_profit(sc, sp, s0, f_call_ask, f_put_ask, f_put_bid, probs, lb_rtn, ub_rtn, p0, s0_ratio):
    cost = compute_cost(sc, sp, s0, f_call_ask, f_put_ask, f_put_bid, s0_ratio)

    drtn = (ub_rtn - lb_rtn) / len(probs)
    exp_rtn = 0.
    for i in range(len(probs)):
        rtn = (i + .5) * drtn + lb_rtn
        price = p0 * (1. + rtn)
        revenue = compute_revenue(price, sc, sp, s0, s0_ratio)
        rr = (revenue - cost) / (cost + s0 * s0_ratio)
        exp_rtn += rr * probs[i]

    return exp_rtn


def calibrate_call_put_asks(p0, lb_rtn, ub_rtn, probs, f_call_ask, f_put_ask, f_put_bid, strike_bounds, s0_ratio):
    def obj_func(x, params):
        sc, sp, s0 = x
        # if sc > sp:
        #     return 9999
        _p0, _f_call_ask, _f_put_ask, _f_put_bid, _lb_rtn, _ub_rtn, _probs, _s0_ratio = params
        exp_rtn = compute_expected_profit(sc, sp, s0, _f_call_ask, _f_put_ask, _f_put_bid, _probs, _lb_rtn, _ub_rtn, _p0, _s0_ratio)
        cost = compute_cost(sc,sp,s0,_f_call_ask,_f_put_ask,_f_put_bid,_s0_ratio)
        return -exp_rtn + abs(cost)

    if strike_bounds is not None:
        bounds = [strike_bounds, strike_bounds, strike_bounds]
    max_rtn = -999
    opt_x = [0, 0]
    for i in range(10):
        s1 = np.random.random() * (strike_bounds[1] - strike_bounds[0]) + strike_bounds[0]
        s2 = np.random.random() * (strike_bounds[1] - strike_bounds[0]) + strike_bounds[0]
        s3 = np.random.random() * (strike_bounds[1] - strike_bounds[0]) + strike_bounds[0]
        x0 = np.array([s1, s2, s3])
        res = minimize(obj_func, x0, args=((p0, f_call_ask, f_put_ask, f_put_bid, lb_rtn, ub_rtn, probs, s0_ratio),),
                       bounds=bounds)
        cost = compute_cost(res.x[0], res.x[1], res.x[2], f_call_ask, f_put_ask, f_put_bid, s0_ratio)
        print(f"optimal sc,sp,s0: {res.x}")
        print(f"maximized expected return: {-res.fun:.2f}")

        if -res.fun > max_rtn :
            max_rtn = -res.fun
            opt_x = res.x

    return opt_x, max_rtn


def prepare_options(sym, exp_date):
    options = get_option_chain_alpha_vantage(sym)
    print(f"{len(options)} options downloaded")
    puts = []
    calls = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'put':
            puts.append(opt)
        if opt['expiration'] == exp_date and opt['type'] == 'call':
            calls.append(opt)

    print(f"{len(puts)} calls returned")
    # for call in calls:
    #     plt.plot(float(call['strike']),float(call['bid']),'.')
    # plt.show()
    return puts, calls


def create_premium_cal(calls, puts, p0):
    strikes = []
    asks = []
    for optn in calls:
        strikes.append(float(optn['strike']))
        asks.append(float(optn['ask']))
    strikes = np.array(strikes)
    asks = np.array(asks)

    call_ask = interp1d(strikes, asks, kind='linear')

    strikes = []
    asks = []
    for optn in puts:
        strikes.append(float(optn['strike']))
        asks.append(float(optn['ask']))
    strikes = np.array(strikes)
    asks = np.array(asks)

    put_ask = interp1d(strikes, asks, kind='linear')

    strikes = []
    asks = []
    for optn in puts:
        strikes.append(float(optn['strike']))
        asks.append(float(optn['bid']))
    strikes = np.array(strikes)
    asks = np.array(asks)

    put_bid = interp1d(strikes, asks, kind='linear')

    bounds = [np.min(strikes), np.max(strikes)]
    # ask_cal = lambda x: f_ask(x) if x <= bounds[1] else  99990.
    # ask_cal = lambda x: 0 if x <= bounds[0] elif x>=bounds[1] x-p0 else f_ask(x)
    f_call_ask = lambda x: 0.05 if x >= bounds[1] else (p0 - x if x <= bounds[0] else call_ask(x))
    f_put_ask = lambda x: 0.05 if x <= bounds[0] else (x - p0 if x >= bounds[1] else put_ask(x))
    f_put_bid = lambda x: 0.05 if x <= bounds[0] else (x - p0 if x >= bounds[1] else put_bid(x))

    return f_call_ask, f_put_ask, f_put_bid, bounds


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
    df, bars_per_day = download_from_yfinance(ticker, period='2y')

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    p0 = df['Close'].values[-1][0]
    print(f"Latest price: {p0}")
    puts, calls = prepare_options(ticker, exp_date)
    f_call_ask, f_put_ask, f_put_bid, strike_bounds = create_premium_cal(calls, puts, p0)

    # lookback_days, min_diff = findBestLookbackDays(22 * 12, 22 * 22, fwd_days, bars_per_day, rtns)
    # print(f"optimal days: {lookback_days}, min_diff: {min_diff}")
    #
    # picked_rtns = rtns[-lookback_days * bars_per_day:]
    #
    # # probs = compute_steady_dist(picked_rtns,lb_rtn=-0.5,ub_rtn=1.)
    # cdfcal = ECDFCal(picked_rtns)
    # probs = compMultiStepProb(fwd_days * bars_per_day, lb_rtn=lb_rtn, ub_rtn=ub_rtn, cdf_cal=cdfcal)
    # plt.plot(probs, '.')

    print(f"n_intervals: {len(rtns) // (fwd_days * bars_per_day)}")
    err = sliding_cdf_error(rtns, fwd_days * bars_per_day, [0.3333, 0.3333, .3333])
    print(f"sliding cdf error: {err:.4f}")
    wts = calibrate_weights(rtns, fwd_days * bars_per_day, nvar=3)

    print(f"Calibrating strike against recent weighted-sum distribution")
    cdfcal = WeightedCDFCal(rtns, wts, fwd_days * bars_per_day)
    probs = compMultiStepProb(fwd_days * bars_per_day, lb_rtn=lb_rtn, ub_rtn=ub_rtn, cdf_cal=cdfcal)
    drtn = (ub_rtn - lb_rtn) / len(probs)
    plt.plot(probs, '.')

    strike_bounds[0] = max([p0 * (1 + lb_rtn), strike_bounds[0]])
    strike_bounds[1] = min([p0 * (1 + ub_rtn), strike_bounds[1]])

    res, exp_rtn = calibrate_call_put_asks(p0, lb_rtn=lb_rtn, ub_rtn=ub_rtn, probs=probs, f_call_ask=f_call_ask,
                                           f_put_ask=f_put_ask, f_put_bid= f_put_bid,
                                           strike_bounds=strike_bounds, s0_ratio=2)

    sc, sp, s0 = res
    print(f"optimal to-sell s_put: {s0:.2f}, to-buy s_call: {sc:.2f}, optimal s_put: {sp:.2f}")
    print(f"premium: {f_put_bid(s0):.2f}, {f_call_ask(sc):.2f}, {f_put_ask(sp):.2f}")
    # breakpoint()
    cost = compute_cost(sc, sp, s0, f_call_ask, f_put_ask, f_put_bid, s0_ratio=2)
    print(f"position cost: {cost:.2f}")
    print(f"expected return: {exp_rtn:.4f}")
    print(
        f"max daily return: {exp_rtn / fwd_days:.4f}, annual return: {exp_rtn / fwd_days * 252:.4f}")

    idx_sc = int((sc / p0 - 1 - lb_rtn) / drtn)
    idx_sp = int((sp / p0 - 1 - lb_rtn) / drtn)
    pb_sc = sum(probs[idx_sc:])
    pb_sp = sum(probs[:idx_sp])
    print(f"profit prob of call,put : {pb_sc:.3f}, {pb_sp:.3f}")

    plt.show()
