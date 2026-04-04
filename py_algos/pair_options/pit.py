import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.environ['ATHENA_HOME'] + "/py_algos/pair_options")

from mkv_cal import compute_total_return_distribution, ECDFCal
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar


def extend_rtns(df, bars_per_day):
    # breakpoint()
    opens = df['Open'].values.ravel()
    highs = df['High'].values.ravel()
    lows = df['Low'].values.ravel()
    closes = df['Close'].values.ravel()
    pcs = np.zeros(len(df) * 3)
    k = 0
    # breakpoint()
    for i in range(len(opens)):
        pcs[k] = opens[i]
        k += 1
        pcs[k] = (highs[i] + lows[i]) * .5
        k += 1
        pcs[k] = closes[i]
        k += 1
    # pdb.set_trace()
    rtns = np.zeros(len(df) * 3)
    for i in range(1, len(pcs)):
        rtns[i - 1] = (pcs[i] - pcs[i - 1]) / pcs[i - 1]

    # rtns = rtns[bars_per_day*3:]
    return rtns, bars_per_day * 3


def model_forecast(df, tid, bars_per_day, lookback_days, fwd_days, vol_scaler):
    past_df = df.iloc[tid - bars_per_day * lookback_days:tid]
    rtns, new_bpd = extend_rtns(past_df, bars_per_day)
    tot_rtns = compute_total_return_distribution(rtns, bars_per_day=new_bpd, lookback_days=lookback_days,
                                                 fwd_days=fwd_days, vol_scaler=vol_scaler)
    # cdf_cal = ECDFCal(tot_rtns)
    return tot_rtns


def process_single_step(i, df, bars_per_day, lookback_days, fwd_days, vol_scaler):
    """
    Worker function for a single PIT calculation.
    """
    # 1. Calculate the target return (actual_y)
    if not df.index[i].dayofweek == 0:
        return -1
    curr_price = df['Close'].iloc[i]
    target_idx = i + fwd_days * bars_per_day
    actual_y = df['Close'].iloc[target_idx] / curr_price - 1

    # 2. Call your forecasting model
    # Note: Ensure model_forecast is thread-safe and doesn't modify the global df
    forecast_rtns = model_forecast(df, i, bars_per_day, lookback_days, fwd_days, vol_scaler)

    # 3. Calculate CDF and PIT
    cdf_cal = ECDFCal(forecast_rtns)
    u_val = cdf_cal.compCDF(actual_y)

    # 4. Clip and return
    return np.clip(u_val, 1e-6, 1 - 1e-6)


def calibrate_garch(df, lookback_days, fwd_days, bars_per_day):
    def obj_func(x, lookback_days, df, fwd_days, bars_per_day, isplot):
        vol_scaler = x
        _, pv = validate_hourly_pit(df, vol_scaler, lookback_days, fwd_days, bars_per_day, isplot)
        print(f"{lookback_days},{vol_scaler}, p-val: {pv}")
        return -pv

    # bounds = [(0, 1)]
    # init = np.array([0.6])
    # res = minimize(obj_func, init,
    #                args=(lookback_days, df, fwd_days, bars_per_day, False),
    #                method="L-BFGS-B",
    #                bounds=bounds,
    #                # options={'eps': 0.01}
    #                )
    res = minimize_scalar(obj_func,bounds=(0.4,.999),method='bounded',
                          args=(lookback_days, df, fwd_days, bars_per_day, False), tol=1e-2)

    vol_scaler = res.x
    # normalized_days = res.x[1]
    print(f"Best var: vol_scaler: {vol_scaler} lookback_days: {lookback_days}")
    validate_hourly_pit(df, vol_scaler, lookback_days, fwd_days, bars_per_day)


def validate_hourly_pit(df, vol_scaler, lookback_days, fwd_days=10, bars_per_day=7, isplot=True):
    """
    Validates the probability distribution forecast using the PIT method.
    """
    sample_size = len(df)
    print(f"Running PIT calculation on {sample_size} samples...")

    start_idx = lookback_days * bars_per_day
    end_idx = len(df) - fwd_days * bars_per_day
    step = bars_per_day

    indices = range(start_idx, end_idx, step)
    pit_values = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_single_step)(i, df, bars_per_day, lookback_days, fwd_days, vol_scaler)
        for i in indices
    )
    pit_values = [x for x in pit_values if x >= 0]

    pit_values = np.array(pit_values)
    print(f"sample size: {len(pit_values)}")

    # 4. Statistical Tests
    # Kolmogorov-Smirnov test for Uniformity [0, 1]
    ks_stat, p_value = stats.kstest(pit_values, 'uniform')

    # 5. Diagnostic Visualization
    if isplot:
        plot_results(
            pit_values, ks_stat, p_value, fwd_days)

    return pit_values, p_value


def plot_results(pit_values, ks_stat, p_value, fwd_days):
    plt.figure(figsize=(10, 6))

    # Histogram should be flat if the model is well-calibrated
    plt.hist(pit_values, bins=30, density=True, alpha=0.6,
             color='#2ca02c', edgecolor='white', label='PIT Empirical')

    # Reference line for perfect calibration
    plt.axhline(1.0, color='red', linestyle='--', lw=2, label='Ideal Uniform')

    plt.title(f"PIT Diagnostic: (Hourly)\n"
              f"Forecast Horizon: {fwd_days}day | KS p-value: {p_value:.4E}")
    plt.xlabel("Cumulative Probability (u)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pit.py <ticker>, [lookback_days=400]")
        sys.exit(1)

    ticker = sys.argv[1]
    lookback_days = 400
    if len(sys.argv) > 2:
        lookback_days = int(sys.argv[2])
    # 1. Download hourly data
    print(f"Downloading hourly data for {ticker}...")
    df = yf.download(ticker, period="730d", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(level=1)

    if df.empty:
        print("Error: No data retrieved.")
        sys.exit(1)

    # validate_hourly_pit(df, vol_scaler=0.7, lookback_days=300, fwd_days=5)
    calibrate_garch(df, lookback_days=lookback_days, fwd_days=5, bars_per_day=7)
