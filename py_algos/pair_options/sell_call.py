import matplotlib.pyplot as plt

from cal_prob import findBestLookbackDays, prepare_rtns
from covered_call import sliding_cdf_error, calibrate_weights
from mkv_cal import *
from option_chain import *
from utils import *


def compExpectedReturn(cur_price, strike, premium, probs, drtn,lb_rtn):
    expected_rtn = 0.

    for i in range(len(probs)):
        r = lb_rtn + (i+0.5)*drtn
        p = cur_price*(1+r)
        if p >= strike:
            rev = premium + strike - cur_price
            rtn = rev/cur_price
        else:
            rtn = (premium )/cur_price
        expected_rtn = expected_rtn + probs[i]*rtn
    return expected_rtn


def prepare_calls(sym,exp_date):
    options = get_option_chain_alpha_vantage(sym)
    print(f"{len(options)} options downloaded")
    calls = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'call':
            calls.append(opt)

    print(f"{len(calls)} calls returned")
    return calls

def calibrate_strike_call(cur_price, calls, rtns, steps, lb_rtn , ub_rtn, cdf_cal ):
    max_rtn = -999990.0
    best_strike = 0.0

    probs = compMultiStepProb(rtns,steps,lb_rtn,ub_rtn, cdf_cal)

    x = np.linspace(lb_rtn,ub_rtn,len(probs))
    plt.plot(x,probs,'.')
    # plt.show()
    # pdb.set_trace()
    drtn = (ub_rtn - lb_rtn) / len(probs)
    for call in calls:
        strike = float(call['strike'])

        # strike_rtn = strike/cur_price - 1.
        # if strike_rtn >= ub_rtn or strike_rtn <= lb_rtn:
        #     continue
        premium = float(call['bid'])
        exp_rtn = compExpectedReturn(cur_price,strike,premium,probs,drtn,lb_rtn)
        # pdb.set_trace()
        idx = int(((strike/cur_price-1.)-lb_rtn)/drtn)
        assign_prob = np.sum(probs[idx+1:])
        print(f"strike: {strike}, asgn prob: {assign_prob:.3f}, exp_rtn: {exp_rtn:.4f}, bid: {premium}, rtn*prob: {(1-assign_prob)*premium/strike*100:.2f}")
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
    df, bars_per_day = download_from_yfinance(ticker, period='730d', interval='1h')

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    cur_price = df['Close'].values[-1][0]

    lookback_days, min_diff = findBestLookbackDays(22 * 6, 730, fwd_days, bars_per_day, rtns)
    print(f"optimal days: {lookback_days}, min_diff: {min_diff}")
    spacing = lookback_days*bars_per_day

    pick_rtns = rtns[-spacing:]

    puts = prepare_calls(ticker,exp_date)
    # pdb.set_trace()

    steps = fwd_days*bars_per_day
    cdf_cal = ECDFCal(pick_rtns)
    best_strike,max_rtn = calibrate_strike_call(cur_price,puts,pick_rtns,steps,lb_rtn = -0.6, ub_rtn = 1.,cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}, exp_profit: {best_strike*max_rtn:.2f}")
    print(f"max daily return: {max_rtn/fwd_days:.4f}, annual return: {max_rtn/fwd_days*252:.4f}")

    print(f"n_intervals: {len(rtns) // (fwd_days * bars_per_day)}")
    err = sliding_cdf_error(rtns, fwd_days * bars_per_day, [0.3333, 0.3333, .3333])
    print(f"sliding cdf error: {err:.4f}")
    wts = calibrate_weights(rtns, fwd_days * bars_per_day, nvar=3)

    cdf_cal = WeightedCDFCal(rtns,wts,fwd_days*bars_per_day)
    best_strike, max_rtn = calibrate_strike_call(cur_price, puts, pick_rtns, steps, lb_rtn=-0.6, ub_rtn=1.,
                                                cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}, exp_profit: {best_strike * max_rtn:.2f}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")
    plt.show()