import numpy as np
import pandas as pd
import statsmodels.sandbox.distributions.gof_new

from mkv_cal import *
from option_chain import *
from utils import *
from arch import arch_model
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ======================================================
# 1️⃣ Load your data here
# r = daily returns
# x = realized variance (from intraday data)
# ======================================================

# Example: simulate data (replace with real data)
np.random.seed(0)
T = 1200

true_params = [-0.2, 0.94, 0.04, 0.1, 0.9, -0.1, 0.2]
omega, beta, gamma, xi, phi, tau, sigma_u = true_params

logh = np.zeros(T)
h = np.zeros(T)
r = np.zeros(T)
x = np.zeros(T)

logh[0] = -1.0
x[0] = 0.1

for t in range(1, T):
    logh[t] = omega + beta * logh[t - 1] + gamma * np.log(x[t - 1])
    h[t] = np.exp(logh[t])

    z = np.random.normal()
    r[t] = np.sqrt(h[t]) * z

    u = np.random.normal(scale=sigma_u)
    x[t] = np.exp(xi + phi * logh[t] + tau * z + u)

x = np.maximum(x, 1e-8)


# ======================================================
# 2️⃣ Log-likelihood function
# ======================================================

def realized_garch_loglik(params, r, x):
    omega, beta, gamma, xi, phi, tau, sigma_u = params

    T = len(r)
    logh = np.zeros(T)
    logh[0] = np.log(np.var(r))

    ll = 0.0

    for t in range(1, T):
        logh[t] = omega + beta * logh[t - 1] + gamma * np.log(x[t - 1])
        h_t = np.exp(logh[t])

        z = r[t] / np.sqrt(h_t)

        # return density
        ll += -0.5 * (np.log(2 * np.pi) + np.log(h_t) + z ** 2)

        # measurement density
        meas = np.log(x[t]) - (xi + phi * logh[t] + tau * z)
        ll += -0.5 * (np.log(2 * np.pi * sigma_u ** 2) + meas ** 2 / sigma_u ** 2)

    return -ll


# ======================================================
# 3️⃣ Estimate parameters
# ======================================================

init = [-0.1, 0.9, 0.05, 0.1, 0.8, 0.0, 0.2]

bounds = [
    (-5, 5),  # omega
    (0.0, 0.999),  # beta
    (0.0, 1.0),  # gamma
    (-5, 5),  # xi
    (0.1, 5),  # phi
    (-5, 5),  # tau
    (1e-6, 5)  # sigma_u
]

res = minimize(realized_garch_loglik, init,
               args=(r, x),
               method="L-BFGS-B",
               bounds=bounds)

params_hat = res.x
print("Estimated parameters:")
print(params_hat)

# ======================================================
# 4️⃣ Filter conditional variance
# ======================================================

omega, beta, gamma, xi, phi, tau, sigma_u = params_hat

logh_filt = np.zeros(T)
logh_filt[0] = np.log(np.var(r))

for t in range(1, T):
    logh_filt[t] = omega + beta * logh_filt[t - 1] + gamma * np.log(x[t - 1])

h_filt = np.exp(logh_filt)


# ======================================================
# 5️⃣ Future Forecasts
# ======================================================

def forecast_realized_garch(logh_T, x_T, params, steps=10):
    omega, beta, gamma, xi, phi, tau, sigma_u = params

    forecasts = []
    logh_next = omega + beta * logh_T + gamma * np.log(x_T)

    for _ in range(steps):
        # expected log realized variance
        logx_next = xi + phi * logh_next

        logh_next = omega + beta * logh_next + gamma * logx_next
        forecasts.append(np.exp(logh_next))

    return np.array(forecasts)


future_vol = forecast_realized_garch(
    logh_filt[-1],
    x[-1],
    params_hat,
    steps=20
)

print("\n20-step ahead variance forecasts:")
print(future_vol)

# ======================================================
# 6️⃣ Plot
# ======================================================

plt.figure()
plt.plot(h_filt, label="Filtered Variance")
plt.plot(range(T, T + 20), future_vol, label="Forecast", linestyle="--")
plt.legend()
plt.title("Realized GARCH Volatility Forecast")
plt.show()


def compute_total_return_distribution(rtns,bars_per_day,fwd_days):
    num_days = len(rtns) // bars_per_day
    daily_matrix = rtns[:num_days * bars_per_day].reshape(num_days, bars_per_day)
    daily_returns = np.sum(daily_matrix, axis=1)
    daily_rv = np.sum(daily_matrix ** 2, axis=1)
    # breakpoint()
    daily_vol = np.sqrt(daily_rv)
    residuals = (daily_returns )/daily_vol
    print(stats.ks_2samp(daily_returns[-100:], daily_returns[:-100]))
    print(stats.ks_2samp(residuals[-100:], residuals[:-100]))

    am = arch_model(daily_returns*100,
                    vol='GARCH',
                    p=1,q=1,
                    # x=np.log(daily_rv),
                    dist='t')
    res = am.fit(disp='off')
    print(res.summary())
    forecasts = res.forecast(horizon=fwd_days)

    # Extract predicted daily variance and convert to Volatility (Standard Deviation)
    forecasted_var = forecasts.variance.iloc[-1]
    forecasted_vol = np.sqrt(forecasted_var)
    breakpoint()



def prepare_rtns(df, tid, lookback, ticker):
    if tid == -1:
        tid = len(df)-1
    k = 0
    pc = np.zeros(lookback * 3)
    for i in range(tid - lookback, tid):
        pc[k] = df['Open'][ticker][i]
        k = k + 1
        pc[k] = (df['High'][ticker][i] + df['Low'][ticker][i]) / 2
        k = k + 1
        pc[k] = df['Close'][ticker][i]
        k = k + 1
    rtns = np.zeros(lookback* 3 )
    for i in range(1, len(pc)):
        rtns[i - 1] = (pc[i] - pc[i - 1]) / pc[i - 1]
    # breakpoint()
    return rtns


def compute_rtn_each_trade(tid, ticker, df, lookback, horizon, asgn_thd, calls, lb_rtn=-0.5, ub_rtn=1.):
    # breakpoint()
    rtns = prepare_rtns(df, tid, lookback,ticker)
    cdf_cal = ECDFCal(rtns)
    probs = compMultiStepProb(horizon, lb_rtn, ub_rtn, cdf_cal)

    drtn = (ub_rtn - lb_rtn) / len(probs)
    picked_strike_rtn = 0
    picked_bid = 0
    for call in calls:
        s_rtn = call['rtn']
        if s_rtn < lb_rtn:
            continue
        if s_rtn > ub_rtn:
            breakpoint()
        idx = (s_rtn - lb_rtn) // drtn
        idx = int(idx)
        if idx < 0:
            breakpoint()
        pb_asgn = sum(probs[idx:])
        if pb_asgn <= asgn_thd:
            picked_strike_rtn = s_rtn
            picked_bid = float(call['bid'])
            break

    # breakpoint()
    pos_price = df.iloc[tid]['Close'][ticker]
    strike = pos_price * (1 + picked_strike_rtn)

    exp_price = df.iloc[tid + horizon]['Close'][ticker]

    if exp_price > strike:
        profit = strike - exp_price + picked_bid
    else:
        profit = picked_bid
    print(f"{df.index[tid]}, {profit:.2f}")
    return profit / pos_price


def back_test(ticker, df, fwd_days, bars_per_day, lookback_days, calls, year=2025):
    lookback = lookback_days * bars_per_day
    horizon = fwd_days * bars_per_day
    # breakpoint()

    # 1. Get all entries that fall on a Monday
    mondays_all = df[(df.index.weekday == 0) & (df.index.year == year)]

    # 2. Group by date and take the first row.
    # .apply(lambda x: x.index[0]) extracts the ACTUAL timestamp label from the original DF.
    monday_timestamps = mondays_all.groupby(mondays_all.index.date).apply(lambda x: x.index[0])

    # 3. Now get_loc will work because the labels are identical
    tids = [df.index.get_loc(ts) for ts in monday_timestamps]
    rtns = []
    for tid in tids:
        # breakpoint()
        r = compute_rtn_each_trade(tid, ticker, df, lookback, horizon, 0.2, calls)
        rtns.append(r)

    print(f"average rtn: {np.mean(rtns)}")
    return rtns


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ticker> <expiration_date>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    exp_date = sys.argv[2]
    cur_price = float(rh.stocks.get_latest_price(ticker)[0])
    print(f"Latest price: {cur_price:.2f}")

    fwd_days = TradeDaysCounter().countTradeDays(exp_date)
    print(f"trading days: {fwd_days}")

    df, bars_per_day = download_from_yfinance(ticker)

    calls, puts = prepare_callsputs(ticker, exp_date)
    for call in calls:
        call['rtn'] = float(call['strike']) / cur_price - 1.

    # back_test(ticker, df, fwd_days, bars_per_day, lookback_days=220, calls=calls)

    rtns = prepare_rtns(df,-1,700*bars_per_day,ticker)
    bars_per_day = bars_per_day * 3
    compute_total_return_distribution(rtns,bars_per_day,fwd_days)





