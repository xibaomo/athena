import os,sys
import pdb
from datetime import datetime,timedelta
import yfinance as yf
from scipy import stats
from cal_prob import findBestLookbackDays,prepare_rtns
from utils import *
from mkv_cal import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import requests


def compExpectedReturn(cur_price, strike, premium, probs, drtn,lb_rtn=-.5):
    expected_rtn = 0.

    for i in range(len(probs)):
        r = lb_rtn + (i+0.5)*drtn
        p = cur_price*(1+r)
        if p >= strike:
            rtn = premium/strike
        else:
            rtn = (premium - strike + p)/strike
        expected_rtn = expected_rtn + probs[i]*rtn
    return expected_rtn


def prepare_puts(sym,exp_date):
    url = 'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=' + sym.upper() + "&apikey=A4L0CXXLQHSWW8ZS"
    r = requests.get(url)
    data = r.json()
    # if not 'data' in data.keys():
    #     pdb.set_trace()
    options = data['data']
    print(f"{len(options)} options downloaded")
    puts = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'put':
            puts.append(opt)

    print(f"{len(puts)} puts returned")
    return puts

def calibrate_strike_put(cur_price, puts, rtns, steps, lb_rtn = -0.5, ub_rtn = 0.5):
    max_rtn = 0.0
    best_strike = 0.0

    probs = compMultiStepProb(rtns,steps)
    plt.plot(probs,'.')
    plt.show()
    # pdb.set_trace()
    drtn = (ub_rtn - lb_rtn) / len(probs)
    for put in puts:
        strike = float(put['strike'])
        # if strike == 275:
        #     pdb.set_trace()
        strike_rtn = strike/cur_price - 1.
        if strike_rtn >= ub_rtn or strike_rtn <= lb_rtn:
            continue
        premium = float(put['bid'])
        exp_rtn = compExpectedReturn(cur_price,strike,premium,probs,drtn,lb_rtn)
        # pdb.set_trace()
        idx = int(((strike/cur_price-1.)-lb_rtn)/drtn)
        assign_prob = np.sum(probs[:idx+1])
        print(f"strike: {strike}, assign prob: {assign_prob:.3f}, exp_rtn: {exp_rtn:.4f}, market value: {premium}")
        if exp_rtn > max_rtn:
            max_rtn = exp_rtn
            best_strike = strike
    return best_strike, max_rtn

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <expiration_date>")
        sys.exit(0)

    ticker = sys.argv[1]
    exp_date = sys.argv[2]

    fwd_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {fwd_days}")
    df, bars_per_day = download_from_yfinance(ticker, period='2y')

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    cur_price = df['Close'].values[-1][0]

    print(f"Latest price: {cur_price}")

    ave_d = eval_stability(rtns)
    print(f"Stability: {ave_d:.5f}")

    lookback_days, min_diff = findBestLookbackDays(22 * 12, 22 * 22, fwd_days, bars_per_day, rtns)
    print(f"optimal days: {lookback_days}, min_diff: {min_diff}")

    pick_rtns = rtns[-lookback_days * bars_per_day:]

    puts = prepare_puts(ticker,exp_date)
    # pdb.set_trace()

    steps = fwd_days*bars_per_day
    best_strike,max_rtn = calibrate_strike_put(cur_price,puts,pick_rtns,steps,lb_rtn = -0.75, ub_rtn = 0.75)
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}")
    print(f"max daily return: {max_rtn/fwd_days:.4f}, annual return: {max_rtn/fwd_days*252:.4f}")