import os,sys
import pdb
from datetime import datetime,timedelta
import yfinance as yf
from scipy import stats
from cal_prob import findBestLookbackDays,prepare_rtns
from utils import *
from mkv_cal import *
from scipy.optimize import minimize
import requests
from option_chain import *
import pdb
import matplotlib.pyplot as plt
# Login to Robinhood
import os
username = os.getenv("BROKER_USERNAME")
password = os.getenv("BROKER_PASSWD")
rh.login(username, password, store_session=True)

def __prepare_calls(sym,exp_date):
    ticker = yf.Ticker(sym)
    chain = ticker.option_chain(exp_date)
    calls = chain.calls
    calls = calls.sort_values(by='strike')
    print(f"count of options: {len(calls)}. max strike: {calls['strike'].values[-1]:.2f}")

    return calls

def prepare_calls(sym,exp_date):
    # url = 'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=' + sym.upper() + "&apikey=A4L0CXXLQHSWW8ZS"
    # r = requests.get(url)
    # data = r.json()
    # # if not 'data' in data.keys():
    # #     pdb.set_trace()
    # options = data['data']
    options = get_option_chain_alpha_vantage(sym)
    print(f"{len(options)} options downloaded")
    calls = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'call':
            calls.append(opt)

    print(f"{len(calls)} calls returned")
    # for call in calls:
    #     plt.plot(float(call['strike']),float(call['bid']),'.')
    # plt.show()
    return calls

def bisection_minimize(f, a, b, tol=1e-5, max_iter=100):
    """
    Minimizes a unimodal function f in the interval [a, b]
    using a bisection-style interval reduction method.
    """
    iter_count = 0
    while (b - a) > tol and iter_count < max_iter:
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if f(m1) < f(m2):
            b = m2
        else:
            a = m1
        iter_count += 1
    x_min = (a + b) / 2
    return x_min, f(x_min)
def __maximize_expected_revenue(rtns, fwd_steps,cur_price, calls):
    def obj_func(xs):
        ub_rtn = xs/cur_price - 1.
        pu,pd = compProb1stHitBounds(rtns,fwd_steps,ub_rtn=ub_rtn,lb_rtn=-.5)
        rev = xs - cur_price
        exp_rev = rev * pu
        print(f"strike: {xs:.2f}, exp_rev: {exp_rev:.2f}")
        return -exp_rev

    opt_s, opt_rev=bisection_minimize(obj_func,cur_price*1.01,cur_price*1.25,tol=1.)

    print(f"optimal strike: {opt_s:.2f}, max expected rev: {-opt_rev:.2f}")
    return -opt_rev
def calibrate_strike(ticker,rtns,fwd_steps, cdf_type, cost, calls):
    max_rev = -9999.
    best_strike = 0.
    cur_price = float(rh.stocks.get_latest_price(ticker)[0])
    for i in range(len(calls)):
        call = calls[i]
        # pdb.set_trace()
        s = float(call['strike'])
        if s < cur_price:
            continue
        ub_rtn = s / cur_price - 1.
        # pdb.set_trace()
        pu, pd = compProb1stHitBounds(rtns, fwd_steps, cdf_type = cdf_type, ub_rtn=ub_rtn, lb_rtn=-.5)
        if pu < 0.05:
            break

        bid = float(call['bid'])
        rev = s - cost + bid
        exp_rev = rev*pu + bid*(1-pu)
        print(f"strike: {s:.2f}, assign prob: {pu:.3f}, exp profit: {exp_rev:.2f}, market value: {bid:.2f}")
        if exp_rev > max_rev:
            max_rev = exp_rev
            best_strike = s
    return best_strike,max_rev

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <expiration_date> [stock_cost] ")
        sys.exit(1)

    ticker = sys.argv[1]
    exp_date = sys.argv[2]
    cost_price = float(rh.stocks.get_latest_price(ticker)[0])
    print(f"Latest price: {cost_price:.2f}")
    if len(sys.argv) == 4:
        cost_price = float(sys.argv[3])


    fwd_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {fwd_days}")

    df, bars_per_day = download_from_yfinance(ticker, period='2y')

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)

    ave_d = eval_stability(rtns)
    print(f"Stability: {ave_d:.5f}")

    lookback_days,min_diff = findBestLookbackDays(22*12,22*22,fwd_days,bars_per_day,rtns)
    # lookback_days, min_diff = findBestLookbackDays(22 * 1, 22 * 6, fwd_days, bars_per_day, rtns)
    print(f"optimal days: {lookback_days}, min_diff: {min_diff}")

    pick_rtns = rtns[-lookback_days*bars_per_day:]

    calls = prepare_calls(ticker,exp_date)

    print("Calibrating strike against long-term distribution")
    best_strike, max_rev = calibrate_strike(ticker,pick_rtns,fwd_days*bars_per_day, 'emp',cost_price, calls)
    print(f"optimal strike: {best_strike:.2f}, max expected profit: {max_rev:.2f}")
    print(f"max daily return: {max_rev/fwd_days/cost_price:.4f}, annual return: {max_rev/fwd_days/cost_price*252:.4f}")

    print(f"Calibrating strike against recent weighted-sum distribution")
    best_strike, max_rev = calibrate_strike(ticker, pick_rtns, fwd_days * bars_per_day, 'wts', cost_price, calls)
    print(f"optimal strike: {best_strike:.2f}, max expected profit: {max_rev:.2f}")
    print(f"max daily return: {max_rev / fwd_days / cost_price:.4f}, annual return: {max_rev / fwd_days / cost_price * 252:.4f}")