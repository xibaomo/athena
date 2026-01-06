import matplotlib.pyplot as plt

from cal_prob import findBestLookbackDays, prepare_rtns
from covered_call import sliding_cdf_error, calibrate_weights
from mkv_cal import *
from option_chain import *
from utils import *
from covered_call import evaluate_latest_wasserstein_distance


def compExpectedReturn(cur_price, strike, premium, probs, drtn,lb_rtn):
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
    options = get_option_chain_alpha_vantage(sym)
    print(f"{len(options)} options downloaded")
    puts = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'put':
            puts.append(opt)

    print(f"{len(puts)} puts returned")
    return puts

def calibrate_strike_put(cur_price, puts, rtns, steps, lb_rtn , ub_rtn, cdf_cal ):
    max_rtn = 0.0
    best_strike = 0.0

    max_profit = -99999
    max_profit_strike = 0.

    probs = compMultiStepProb(steps,lb_rtn,ub_rtn, cdf_cal)

    x = np.linspace(lb_rtn,ub_rtn,len(probs))
    plt.plot(x,probs,'.')
    # plt.show()
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

    # spacing,min_diff = find_stablest_spacing(rtns,22*bars_per_day,2*bars_per_day)
    # print(f"length of rtns: {len(rtns)}, min ave diff: {min_diff}, spacing days: {spacing//bars_per_day}")

    lookback_days, min_diff = findBestLookbackDays(22 * 6, 730, fwd_days, bars_per_day, rtns)
    print(f"optimal days: {lookback_days}, min_diff: {min_diff}")
    spacing = lookback_days*bars_per_day

    pick_rtns = rtns[-spacing:]

    # puts = prepare_puts(ticker,exp_date)
    calls,puts = prepare_callsputs(ticker,exp_date)
    call_put_ratio = call_put_ask_ratio(0.25,calls,puts)
    print(f"0.25_delta call/put ask_ratio: {call_put_ratio:.3f}")
    # pdb.set_trace()

    steps = fwd_days*bars_per_day
    cdf_cal = ECDFCal(pick_rtns)
    best_strike,max_rtn = calibrate_strike_put(cur_price,puts,pick_rtns,steps,lb_rtn = -0.6, ub_rtn = 1.,cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}, exp_profit: {best_strike*max_rtn:.2f}")
    print(f"max daily return: {max_rtn/fwd_days:.4f}, annual return: {max_rtn/fwd_days*252:.4f}")

    print(f"searching for best lookback days...")
    m_range = range(fwd_days * bars_per_day, 22 * 5 * bars_per_day)
    res = find_best_m_given_n(rtns, fwd_days * bars_per_day, m_range)
    print(f"best lookback days: {res['m'] / bars_per_day:.2f}, max corr: {res['corr']:.4f}")
    # breakpoint()

    backdays = round(res['m'] / bars_per_day)
    n_back = res['m']
    horizon = fwd_days * bars_per_day

    dmin = 99999.
    best_n_back = 0
    for i in [1, 2, 3]:
        d = evaluate_latest_wasserstein_distance(rtns, n_back * i, horizon)
        print(f"lookback: {n_back * i}, wasserstein distance: {d:.5f}")
        if d < dmin:
            dmin = d
            best_n_back = i * n_back

    n_back = best_n_back
    print(f"Searching for subarray ({n_back / bars_per_day} days) with the most likely distribution...")

    x = rtns[-n_back:]
    y = rtns[:-n_back]

    res = analog_distribution_forecast(x, y, horizon, K=5)
    pick_rtns = res['future_samples']

    cdf_cal = ECDFCal(pick_rtns)
    best_strike, max_rtn = calibrate_strike_put(cur_price, puts, pick_rtns, steps, lb_rtn=-0.6, ub_rtn=1.,
                                                cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}, exp_profit: {best_strike * max_rtn:.2f}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")
    plt.show()