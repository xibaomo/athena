import pdb
import sys,os
import robin_stocks.robinhood as rh
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from scipy import stats
from mkv_cal import *
from utils import *
from statsmodels.distributions.empirical_distribution import ECDF
# username = os.getenv("BROKER_USERNAME")
# password = os.getenv("BROKER_PASSWD")
# rh.login(username, password)


def __compute_ave_dist_diff(lookback_days, lookfwd_days, bars_per_day, rtns):
    lookback = lookback_days*bars_per_day
    lookfwd = lookfwd_days*bars_per_day
    ds = []
    for i in np.arange(lookback,len(rtns)-lookfwd,bars_per_day):
        t1 = rtns[i-lookback:i]
        t2 = rtns[i:i+lookfwd]
        d,p = stats.ks_2samp(t1,t2)
        ds.append(d)
    return np.mean(ds)
def compute_latest_dist_diff(lookback_days,lookfwd_days,bars_per_day,rtns):
    lookback = lookback_days * bars_per_day
    lookfwd = lookfwd_days * bars_per_day
    l1 = lookback+lookfwd
    t1 = rtns[-l1:-lookfwd]
    t2 = rtns[-lookfwd:]
    d,p = stats.ks_2samp(t1,t2)
    return d

def verify_prob(rtns, lookback_days, lookfwd_days,bars_per_day, rtn_bound):
    lookback = lookback_days*bars_per_day
    lookfwd  = lookfwd_days*bars_per_day

    dps = []
    for i in np.arange(lookback,len(rtns)-lookfwd,bars_per_day):
        t1 = rtns[i-lookback:i]
        t2 = rtns[i:i+lookfwd]
        pu1,_ = compProb1stHidBounds(t1,lookfwd,rtn_bound,-rtn_bound)
        pu2,_ = compProb1stHidBounds(t2,lookfwd,rtn_bound,-rtn_bound)
        dps.append((pu1-pu2)**2)
        print(f"diff of up probs: {abs(pu1-pu2)}")

    ave_dp = np.sqrt(np.mean(dps))
    print(f"diff of up probs: min: {np.min(dps)}, max: {np.max(dps)}, rms: {ave_dp}")

def calibrate_lookback_days(rtns,lookfwd_days, lookback_days_range,bars_per_day):
    lb,ub = lookback_days_range
    while ub - lb > 1:
        mid = (ub+lb)//2
        ave_dd = compute_latest_dist_diff(mid,lookfwd_days,bars_per_day,rtns)
        ave_dd_plus = compute_latest_dist_diff(mid+1,lookfwd_days,bars_per_day,rtns)

        print(f"lookback days: {mid}, ave_dd: {ave_dd}")
        if ave_dd < ave_dd_plus:
            ub = mid
        else:
            lb = mid
    return ub
def plot_varylookback_distdiff(lb_days,ub_days,lookfwd_days, bars_per_day,rtns):
    xs = np.arange(lb_days,ub_days,5)
    ys = []
    for x in xs:
        y = compute_latest_dist_diff(x,lookfwd_days,bars_per_day,rtns)
        ys.append(y)

    plt.plot(xs,ys,'.-')
    plt.xlabel('lookback days')
    plt.ylabel('Ave dist diff')
    # plt.show(block=False)

def plot_recent_dist_diff(lookback_days, lookfwd_days, bars_per_day, rtns,recent_days=100):
    lookback = lookback_days*bars_per_day
    lookfwd = lookfwd_days*bars_per_day
    rs = rtns[-lookback-recent_days*bars_per_day:]
    ds = []
    for i in np.arange(lookback,len(rs)-lookfwd,bars_per_day):
        t1 = rs[i-lookback:i]
        t2 = rs[i:i+lookfwd]
        d,p = stats.ks_2samp(t1,t2)
        ds.append(d)
    plt.figure()
    plt.plot(ds,'.-')
    plt.xlabel('T-n days')
    plt.ylabel('dist diff')

def prepare_rtns(df,bars_per_day):
    opens = df['Open'].values
    highs = df['High'].values
    lows  = df['Low'].values
    pcs = np.zeros(len(df)*2)
    k=0
    for i in range(len(opens)):
        pcs[k] = opens[i]
        k+=1
        pcs[k] = (highs[i]+lows[i])*.5
        k+=1
    # pdb.set_trace()
    rtns = np.zeros_like(pcs)
    for i in range(1,len(rtns)):
        rtns[i] = (pcs[i]-pcs[i-1])/pcs[i-1]

    rtns = rtns[bars_per_day*2:]
    return rtns,bars_per_day*2

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: {} <ticker> <fwd_days> <rtn_bound> [lookback_days_lb] [lookback_days_ub]".format(sys.argv[0]))
        sys.exit(1)

    ticker = sys.argv[1]
    fwd_days = int(sys.argv[2])
    ub_rtn = float(sys.argv[3])
    lookback_days_lb = 10
    if len(sys.argv) > 4:
        lookback_days_lb = int(sys.argv[4])
    lookback_days_ub = 22*12
    if len(sys.argv) > 5:
        lookback_days_ub = int(sys.argv[5])
    lb_rtn = -ub_rtn
    # df = download_from_robinhood(ticker)
    df,bars_per_day = download_from_yfinance(ticker,period='2y')

    rtns = df['Open'].pct_change().values
    rtns,bars_per_day = prepare_rtns(df,bars_per_day)
    print(f"length of returns: {len(rtns)}, bars per day: {bars_per_day}")
    plot_varylookback_distdiff(10,22*12,fwd_days,bars_per_day,rtns)

    print(f"Calibrating lookback days between [{lookback_days_lb},{lookback_days_ub}]")
    lookback_days = calibrate_lookback_days(rtns,fwd_days,[lookback_days_lb,lookback_days_ub],bars_per_day)
    print(f"optmized lookback days: {lookback_days}")

    ecdf = ECDF(rtns[-fwd_days * bars_per_day:])
    plt.figure()
    plt.plot(ecdf.x, ecdf.y, '.')
    ecdf = ECDF(rtns[-(lookback_days+fwd_days)*bars_per_day:-fwd_days*bars_per_day])
    plt.plot(ecdf.x, ecdf.y, 'r.')
    plot_recent_dist_diff(lookback_days,fwd_days,bars_per_day,rtns)
    # verify_prob(rtns,lookback_days,fwd_days,bars_per_day,ub_rtn)

    steps = fwd_days*bars_per_day
    lookback=lookback_days*bars_per_day
    pu,pd = compProb1stHidBounds(rtns[-lookback:],steps,ub_rtn=ub_rtn,lb_rtn=-ub_rtn)
    print(f"{fwd_days}-day probability of hitting {ub_rtn*100:.2f}%: {pu:.4}, hitting {lb_rtn*100:.2f}%: {pd:.4f}")
    print(f"sum: {pu + pd:.4f}")

    plt.show()
