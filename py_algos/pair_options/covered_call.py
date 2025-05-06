import os,sys
from datetime import datetime,timedelta
import yfinance as yf
from scipy import stats
from cal_prob import findBestLookbackDays,prepare_rtns
from utils import *
from mkv_cal import *
from scipy.optimize import minimize
# Login to Robinhood
username = os.getenv("BROKER_USERNAME")
password = os.getenv("BROKER_PASSWD")
rh.login(username, password)
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
def maximize_expected_revenue(rtns, fwd_steps,cur_price):
    def obj_func(xs):
        ub_rtn = xs/cur_price - 1.
        pu,pd = compProb1stHitBounds(rtns,fwd_steps,ub_rtn=ub_rtn,lb_rtn=-.5)
        rev = xs - cur_price
        exp_rev = rev * pu
        print(f"strike: {xs:.2f}, exp_rev: {exp_rev:.2f}")
        return -exp_rev

    opt_s, opt_rev=bisection_minimize(obj_func,cur_price*1.01,cur_price*1.25,tol=1.)

    print(f"optimal strike: {opt_s}, max expected rev: {-opt_rev:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <expiration_date>  ")
        sys.exit(1)

    ticker = sys.argv[1]
    exp_date = sys.argv[2]
    cur_price = float(rh.stocks.get_latest_price(ticker)[0])
    print(f"Latest price: {cur_price:.2f}")

    fwd_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {fwd_days}")

    df, bars_per_day = download_from_yfinance(ticker, period='2y')

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)

    lookback_days,min_diff = findBestLookbackDays(22,22*22,fwd_days,bars_per_day,rtns)
    print(f"optimal days: {lookback_days}, min_diff: {min_diff}")

    pick_rtns = rtns[-lookback_days*bars_per_day:]
    maximize_expected_revenue(pick_rtns,fwd_days*bars_per_day,cur_price)