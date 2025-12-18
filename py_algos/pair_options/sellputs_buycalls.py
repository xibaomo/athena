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
from covered_call import sliding_cdf_error, calibrate_weights


def compExpectedReturn(cur_price, s_call, s_put, premium, probs, drtn, lb_rtn, ratio):
    expected_rtn = 0.
    max_loss = s_put * ratio  # max_loss is the cash secured

    for i in range(len(probs)):
        r = lb_rtn + (i + 0.5) * drtn
        p = cur_price * (1 + r)
        if p >= s_put:
            rtn = (p - s_call + premium) / max_loss
        elif p > s_call and p < s_put:
            rev = p - s_call - (s_put - p) * ratio + premium
            rtn = rev / max_loss
        else:
            loss = (s_put - p) * ratio
            rtn = (- loss + premium) / max_loss
            # rtn=0
        expected_rtn = expected_rtn + probs[i] * rtn
    return expected_rtn


def find_match_sellputs_buycall(opt_idx, puts, calls, ratio):
    market_value = float(puts[opt_idx]['bid']) * ratio
    for i in range(len(calls)):
        ask = float(calls[i]['ask'])
        if ask <= market_value:
            return i
    # not found
    return -2


def __prepare_options(sym, exp_date):
    options = get_option_chain_alpha_vantage(sym)
    print(f"{len(options)} options downloaded")
    puts = []
    calls = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'put':
            puts.append(opt)
        if opt['expiration'] == exp_date and opt['type'] == 'call':
            calls.append(opt)

    print(f"{len(puts)} calls/puts returned")
    puts = sorted(puts, key=lambda x: float(x["strike"]))
    calls = sorted(calls, key=lambda x: float(x["strike"]))
    return calls, puts


def calibrate_strike_put(cur_price, puts, calls, rtns, steps, lb_rtn, ub_rtn, ratio, cdf_cal):
    max_rtn = -999990.0
    best_strike = []
    best_premium = 0

    probs = compMultiStepProb(steps, lb_rtn, ub_rtn, cdf_cal)

    x = np.linspace(lb_rtn, ub_rtn, len(probs))
    plt.plot(x, probs, '.')
    # plt.show()
    # pdb.set_trace()
    drtn = (ub_rtn - lb_rtn) / len(probs)
    for i in range(len(puts)):
        put = puts[i]
        strike = float(put['strike'])
        bid = float(put['bid'])
        # if strike - bid > cur_price:
        #     continue
        match_idx = find_match_sellputs_buycall(i, puts, calls, ratio)
        if match_idx < 0:
            continue

        premium = bid * ratio - float(calls[match_idx]['ask'])
        # premium = 0
        s_put = strike
        s_call = float(calls[match_idx]['strike'])
        if s_call > s_put:
            continue
        exp_rtn = compExpectedReturn(cur_price, s_call=s_call, s_put=s_put, premium=0, probs=probs, drtn=drtn,
                                     lb_rtn=lb_rtn, ratio=ratio)
        # pdb.set_trace()

        safe_price = (ratio * s_put + s_call) / (ratio + 1)
        print(
            f"strike pair: {strike},{s_call}, exp_rtn: {exp_rtn:.4f}, prem: {put['bid']},{calls[match_idx]['ask']}, {premium:.2f},safe: {safe_price:.2f}")
        if exp_rtn > max_rtn:
            max_rtn = exp_rtn
            best_strike = [s_put, s_call]
            best_premium = premium

    safe_price = (-0 + best_strike[1] + ratio * best_strike[0]) / (ratio + 1)
    return best_strike, max_rtn, safe_price


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
    best_strike, max_rtn, safe_price = calibrate_strike_put(cur_price, puts, calls, pick_rtns, steps,
                                                            lb_rtn=-0.5, ub_rtn=1., ratio=ratio, cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}, safe_price: {safe_price:.2f}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")

    ##calibrate weights
    print(f"n_intervals: {len(rtns) // (fwd_days * bars_per_day)}")
    err = sliding_cdf_error(rtns, fwd_days * bars_per_day, [0.3333, 0.3333, .3333])
    print(f"sliding cdf error: {err:.4f}")
    wts = calibrate_weights(rtns, fwd_days * bars_per_day, nvar=3)

    cdf_cal = WeightedCDFCal(rtns, wts, fwd_days * bars_per_day)
    best_strike, max_rtn, safe_price = calibrate_strike_put(cur_price, puts, calls, pick_rtns, steps,
                                                            lb_rtn=-0.5, ub_rtn=1., ratio=ratio, cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}, safe_price: {safe_price:.2f}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")
    plt.show()
