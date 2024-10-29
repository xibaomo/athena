import sys,os
import robin_stocks.robinhood as rh
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from scipy import stats
from mkv_cal import *
from utils import *
# username = os.getenv("BROKER_USERNAME")
# password = os.getenv("BROKER_PASSWD")
# rh.login(username, password)


def compute_ave_dist_diff(lookback_days, lookfwd_days, bars_per_day, rtns):
    lookback = lookback_days*bars_per_day
    lookfwd = lookfwd_days*bars_per_day

    ds = []
    for i in range(lookback,len(rtns)-lookfwd):
        t1 = rtns[i-lookback:i]
        t2 = rtns[i:i+lookfwd]
        d,p = stats.ks_2samp(t1,t2)
        ds.append(d)
    return np.mean(ds)

def calibrate_lookback_days(rtns,lookfwd_days, lookback_days_range,bars_per_day):
    lb,ub = lookback_days_range
    while ub - lb > 1:
        mid = (ub+lb)//2
        ave_dd = compute_ave_dist_diff(mid,lookfwd_days,bars_per_day,rtns)
        ave_dd_plus = compute_ave_dist_diff(mid+1,lookfwd_days,bars_per_day,rtns)

        print(f"lookback days: {mid}, ave_dd: {ave_dd}")
        if ave_dd < ave_dd_plus:
            ub = mid
        else:
            lb = mid
    return lb
def plot_ave_distdiff(lb_days,ub_days,lookfwd_days, bars_per_day,rtns):
    xs = np.arange(lb_days,ub_days,10)
    ys = []
    for x in xs:
        y = compute_ave_dist_diff(x,lookfwd_days,bars_per_day,rtns)
        ys.append(y)

    plt.plot(xs,ys,'.-')
    plt.xlabel('lookback days')
    plt.ylabel('Ave dist diff')
    plt.show()
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: {} <ticker> <days> <rtn>".format(sys.argv[0]))
        sys.exit(1)

    ticker = sys.argv[1]
    fwd_days = int(sys.argv[2])
    ub_rtn = float(sys.argv[3])
    lb_rtn = -ub_rtn
    # df = download_from_robinhood(ticker)
    df,bars_per_day = download_from_yfinance(ticker)

    rtns = df['Close'].pct_change().values

    plot_ave_distdiff(22,22*10,fwd_days,bars_per_day,rtns)

    lookback_days = calibrate_lookback_days(rtns,fwd_days,[22,22*10],bars_per_day)
    print(f"optmized lookback days: {lookback_days}")

    steps = fwd_days*bars_per_day
    pu,pd = compProb1stHidBounds(ticker,df,steps,lookback=lookback_days*bars_per_day, ub_rtn=ub_rtn,lb_rtn=-ub_rtn)
    print(f"{fwd_days}-day probability of hitting {ub_rtn*100:.2f}%: {pu:.4}, hitting {lb_rtn*100:.2f}%: {pd:.4f}")
    print(f"sum: {pu + pd:.4f}")

