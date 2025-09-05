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
# username = os.getenv("BROKER_USERNAME")
# password = os.getenv("BROKER_PASSWD")
# rh.login(username, password, store_session=True)
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

def interval_means_and_pacf(series, spacing, nlags=20):
    series = np.asarray(series)

    # truncate to make length divisible by spacing
    n = len(series) // spacing * spacing
    # breakpoint()
    reshaped = series[:n].reshape(-1, spacing)
    interval_means = reshaped.mean(axis=1)
    interval_std = reshaped.std(axis=1)

    # pacf_vals = pacf(interval_means, nlags=nlags, method="ywmle")
    return interval_means,interval_std
def compute_empirical_cdf(data, grid):
    """Compute empirical CDF over a fixed grid."""
    cdf = np.searchsorted(np.sort(data), grid, side='right') / len(data)
    return cdf

def sliding_cdf_error(data, spacing, weights):
    nw = len(weights)
    rem = len(data) % spacing
    if rem > 0:
        data = data[rem:]
    n_intervals = (len(data) - spacing) // spacing - nw
    if n_intervals <= 0:
        raise ValueError("Data too short for the given spacing and weights")

    total_error = 0
    pn = 100
    grid = np.linspace(np.min(data), np.max(data), pn)  # CDF evaluation grid

    for i in range(n_intervals):
        intervals = [data[(i + j) * spacing:(i + j + 1) * spacing] for j in range(nw+1)]
        cdfs = [compute_empirical_cdf(interval, grid) for interval in intervals]

        weighted_cdf = sum(w * c for w, c in zip(weights, cdfs[:nw]))
        true_cdf = cdfs[nw]

        error = np.max((weighted_cdf - true_cdf) ** 2)
        total_error += error

    return np.sqrt(total_error/n_intervals)
def calibrate_weights(rtns,spacing, nvar=4):
    def obj_func(wts,params):
        _rtns,_spacing = params
        cost = sliding_cdf_error(_rtns,_spacing,wts)
        return cost

    constraints = {
        'type': 'eq',
        'fun': lambda x: np.sum(x) - 1
    }
    bounds=None
    bounds=[(0,None) for _ in range(nvar)]
    x0 = np.array([1./nvar for _ in range(nvar)])
    res = minimize(obj_func,x0,((rtns,spacing),),bounds=bounds,constraints=constraints)
    print(f"optimal weights: {res.x}")
    print(f"minimized cdf err: {res.fun}")

    return res.x

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
def calibrate_strike(ticker,fwd_steps, cost, calls, cdf_cal):
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

        pu, pd = compProb1stHitBounds(fwd_steps, cdf_cal=cdf_cal, ub_rtn=ub_rtn, lb_rtn=-.5)
        if pu < 0.05:
            break

        bid = float(call['bid'])
        rev = s - cost + bid
        exp_rev = rev*pu + bid*(1-pu)
        print(f"strike: {s:.2f}, asgn prob: {pu:.3f}, exp profit: {exp_rev:.2f}, bid: {bid:.2f}, bid*prob: {(1-pu)*bid:.2f}")
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

    df, bars_per_day = download_from_yfinance(ticker)

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    print(f"rtn range: {np.min(rtns),np.max(rtns)}")

    interval_means, interval_stds = interval_means_and_pacf(rtns, 22*bars_per_day, nlags=10)

    # Plot interval means
    plt.figure(figsize=(10, 4))
    plt.plot(interval_means**1, marker='o')
    plt.title("Interval Means (spacing=5)")
    plt.xlabel("Interval index")
    plt.ylabel("Mean value")
    plt.grid(True)
    # plt.show()

    # Plot PACF
    plt.figure(figsize=(10, 4))
    plot_pacf(interval_means, lags=10, method="ywmle")
    # plot_acf(interval_means, lags=20)
    plt.title("PACF of Interval Means")

    # Plot PACF
    plt.figure(figsize=(10, 4))
    plot_pacf(interval_stds, lags=10, method="ywmle")
    # plot_acf(interval_stds, lags=20)
    plt.title("PACF of Interval std")
    plt.show()
