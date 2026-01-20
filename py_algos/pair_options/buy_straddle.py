import matplotlib.pyplot as plt
from cal_prob import findBestLookbackDays, prepare_rtns
from mkv_cal import *
from option_chain import *
from utils import *

def compExpectedReturn(cur_price, strike, premium, probs, drtn,lb_rtn):
    expected_rtn = 0.

    for i in range(len(probs)):
        r = lb_rtn + (i+0.5)*drtn
        p = cur_price*(1+r)
        if p >= strike:
            rev = p - strike
        else:
            rev = strike - p
        rtn = rev/premium - 1.
        expected_rtn = expected_rtn + probs[i]*rtn
    return expected_rtn

def calibrate_strike_straddle(cur_price, calls, puts, steps, lb_rtn , ub_rtn, cdf_cal ):
    #make a table to lookup call/put ask for a strike
    strike2ask = {}
    for call in calls:
        strike = float(call['strike'])
        strike2ask[strike] = [float(call['ask'])]
    for put in puts:
        strike = float(put['strike'])
        strike2ask[strike].append(float(put['ask']))
    max_rtn = -99990.0
    best_strike = 0.0

    probs = compMultiStepProb(steps,lb_rtn,ub_rtn, cdf_cal)

    x = np.linspace(lb_rtn,ub_rtn,len(probs))
    plt.plot(x,probs,'.')
    # plt.show()
    # pdb.set_trace()
    drtn = (ub_rtn - lb_rtn) / len(probs)
    for strike in strike2ask.keys():
        call_ask,put_ask = strike2ask[strike]

        premium = call_ask+put_ask
        exp_rtn = compExpectedReturn(cur_price,strike,premium,probs,drtn,lb_rtn)
        print(f"strike: {strike}, prem: {call_ask:.2f},{put_ask:.2f}, exp_rtn: {exp_rtn:.2f}")
        # pdb.set_trace()
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

    calls,puts = prepare_callsputs(ticker,exp_date)
    call_put_ratio = call_put_ask_ratio(0.25,calls,puts)
    print(f"0.25_delta call/put ask_ratio: {call_put_ratio:.3f}")
    # pdb.set_trace()

    horizon = fwd_days*bars_per_day
    cdf_cal = ECDFCal(pick_rtns)
    best_strike,max_rtn = calibrate_strike_straddle(cur_price,calls,puts,horizon,lb_rtn = -0.6, ub_rtn = 1.,cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}, exp_profit: {best_strike*max_rtn:.2f}")
    print(f"max daily return: {max_rtn/fwd_days:.4f}, annual return: {max_rtn/fwd_days*252:.4f}")

    cdf_cal = compute_historical_distribution(rtns, fwd_days, bars_per_day)
    best_strike, max_rtn = calibrate_strike_straddle(cur_price, calls, puts, horizon, lb_rtn=-0.6, ub_rtn=1.,
                                                cdf_cal=cdf_cal)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")
    plt.show()