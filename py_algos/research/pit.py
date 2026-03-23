import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.environ['ATHENA_HOME'] + "/py_algos/pair_options")

from mkv_cal import compute_total_return_distribution, ECDFCal


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


def model_forecast( past_df, bars_per_day, lookback_days, fwd_days):
    rtns, new_bpd = extend_rtns(past_df, bars_per_day)
    tot_rtns = compute_total_return_distribution(rtns, bars_per_day=new_bpd, lookback_days=lookback_days,
                                                 fwd_days=fwd_days)
    # cdf_cal = ECDFCal(tot_rtns)
    return tot_rtns


def validate_hourly_pit(df, lookback_days, fwd_days=5, bars_per_day=7):
    """
    Validates the probability distribution forecast using the PIT method.
    """
    # breakpoint()
    sample_size = len(df)
    print(f"Running PIT calculation on {sample_size} samples...")

    # 3. Walk-forward Validation Loop
    # We start after 24 hours of history to allow for volatility calculation

    lk_bars = lookback_days * bars_per_day
    pit_values = []
    for i in range(lookback_days * bars_per_day, sample_size - fwd_days * bars_per_day, bars_per_day):

        curr_price = df['Close'].iloc[i]
        past_df = df.iloc[i - lk_bars:i]
        actual_y = df['Close'].iloc[i+fwd_days*bars_per_day]/curr_price-1

        # Call your forecasting model
        forecast_rtns = model_forecast(past_df, bars_per_day, lookback_days, fwd_days)
        cdf_cal = ECDFCal(forecast_rtns)
        # Calculate PIT value: u = CDF(actual_return)
        u_val = cdf_cal.compCDF(actual_y)
        # breakpoint()

        # Clip to avoid numerical issues at boundaries
        u_val = np.clip(u_val, 1e-6, 1 - 1e-6)
        pit_values.append(u_val)

    pit_values = np.array(pit_values)

    # 4. Statistical Tests
    # Kolmogorov-Smirnov test for Uniformity [0, 1]
    ks_stat, p_value = stats.kstest(pit_values, 'uniform')

    # 5. Diagnostic Visualization
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
              f"Forecast Horizon: {fwd_days}h | KS p-value: {p_value:.4E}")
    plt.xlabel("Cumulative Probability (u)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pit.py <ticker>")
        sys.exit(1)

    ticker = sys.argv[1]
    # 1. Download hourly data
    print(f"Downloading hourly data for {ticker}...")
    df = yf.download(ticker, period="730d", interval="1h")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(level=1)

    if df.empty:
        print("Error: No data retrieved.")
        sys.exit(1)

    validate_hourly_pit(df, 150)
