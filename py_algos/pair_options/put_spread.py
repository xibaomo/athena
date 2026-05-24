import matplotlib.pyplot as plt
from cal_prob import findBestLookbackDays, prepare_rtns
from mkv_cal import *
from option_chain import *
from utils import *

def compExpectedReturn(cur_price, strike_sell, strike_buy, premium, drtn,cdf_cal):
    expected_rtn = 0.
    max_loss = strike_sell-strike_buy - premium
    if max_loss > 10:
        return -999
    # max_loss = 1
    # breakpoint()
    ### 3 cases
    # case 1: p > strike_sell
    rtn = premium/max_loss
    pb = cdf_cal.compRangeProb(strike_sell/cur_price-1.,1.)
    expected_rtn += rtn*pb

    # case 2: p < strike_buy, loss is fixed: strike_buy - strike_sell
    rtn = -1
    pb = cdf_cal.compRangeProb(-1., strike_buy/cur_price-1.)
    expected_rtn += rtn*pb

    # case 3: in the between
    rtn_ub = strike_sell/cur_price - 1.
    rtn_lb = strike_buy/cur_price - 1.
    npb = int((rtn_ub - rtn_lb)/drtn)
    probs = np.zeros(npb)
    for i in range(npb):
        r = (i+.5)*drtn + rtn_lb
        probs[i] = cdf_cal.compRangeProb(r-drtn/2,r+drtn/2)
        price = cur_price*(r+1.)
        loss = strike_sell - price
        rtn = (premium - loss) / max_loss
        expected_rtn += rtn*probs[i]

    return expected_rtn

def calibrate_strike_put_total_rtns(cur_price, puts, tot_rtns ):
    print(f"Calibrating put to sell and put to buy ...")
    max_rtn = -99999
    best_strike = None

    max_profit = -99999
    max_profit_strike = 0.
    cdf_cal = ECDFCal(tot_rtns)
    drtn = 0.001/4

    # x = np.linspace(lb_rtn,ub_rtn,len(probs))
    # plt.plot(x,probs)
    # for put_sell in puts:
    for i in range(len(puts)):
        put_sell = puts[i]
        strike_sell = float(put_sell['strike'])
        for j in range(i-1):
            put_buy = puts[j]
        # for put_buy in puts:
            strike_buy = float(put_buy['strike'])
            if strike_buy >= strike_sell:
                continue
            premium = float(put_sell['bid']) - float(put_buy['ask'])

            exp_rtn = compExpectedReturn(cur_price,strike_sell,strike_buy,premium,drtn,cdf_cal)

            # print(f"strike: {strike}, asgn prob: {assign_prob:.3f}, exp_rtn: {exp_rtn:.4f}, bid: {premium}, rtn*prob: {(1-assign_prob)*premium/strike*100:.2f}")

            if exp_rtn > max_rtn:
                max_rtn = exp_rtn
                best_strike = [strike_sell,strike_buy]
                print(f"strikes: sell {strike_sell:.2f}, buy {strike_buy:.2f}, exp_rtn: {exp_rtn:.4f}")
                print(f"bid: {put_sell['bid']}, ask: {put_buy['ask']}, max_rev: {premium:.2f}, "
                      f"max_loss: {(strike_sell-strike_buy - premium):2f}")

    return best_strike, max_rtn

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <expiration_date> <ticker> <volatility scaler>  ")
        sys.exit(1)

    exp_date = sys.argv[1]
    ticker = sys.argv[2]
    vol_scaler = float(sys.argv[3])

    fwd_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {fwd_days}")
    df, bars_per_day = download_from_yfinance(ticker, period='730d', interval='1h')

    # rtns = df['Open'].pct_change().values
    rtns, bars_per_day = prepare_rtns(df, bars_per_day)
    print(f"length of rtns: {len(rtns)}, bars_per_day: {bars_per_day}")
    cur_price = float(rh.stocks.get_latest_price(ticker)[0])

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


    lookback_days = 300
    tot_rtns = compute_total_return_distribution(rtns, bars_per_day, lookback_days, fwd_days, vol_scaler=vol_scaler)
    best_strike, max_rtn = calibrate_strike_put_total_rtns(cur_price, puts, tot_rtns)
    print(f"Latest price: {cur_price:.2f}")
    print(f"best strike: {best_strike}, max_rtn: {max_rtn}, exp_profit: {best_strike[0] * max_rtn:.2f}")
    print(f"max daily return: {max_rtn / fwd_days:.4f}, annual return: {max_rtn / fwd_days * 252:.4f}")
    print(f"sym: {ticker}, latest price: {cur_price:.2f}")
    plt.show()