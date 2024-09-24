import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from kalman_motion import add_days_to_date
import yfinance as yf
import sys,os

def check_white_noise(x, alpha=0.05):
    """
    Check if a given array is white noise using multiple methods.

    Args:
        x: 1D numpy array (time series).
        alpha: Significance level for Ljung-Box test (default is 0.05).

    Returns:
        results: Dictionary with white noise check results.
    """
    results = {}

    # 1. Plot Autocorrelation Function (ACF)
    # plt.figure(figsize=(10, 4))
    plot_acf(x, lags=20, alpha=0.05)
    plt.title("Autocorrelation Function (ACF)")
    plt.show()

    # 2. Ljung-Box test for white noise (null hypothesis: no autocorrelation)
    lb_stat, lb_p_value = acorr_ljungbox(x, lags=[10], return_df=False)

    # Null hypothesis: The series is white noise (no autocorrelation)
    if lb_p_value < alpha:
        results['Ljung-Box Test'] = f"Not white noise (p-value={lb_p_value[0]:.4f})"
    else:
        results['Ljung-Box Test'] = f"Possibly white noise (p-value={lb_p_value[0]:.4f})"

    # 3. Check the mean and variance
    mean_val = np.mean(x)
    var_val = np.var(x)
    results['Mean'] = mean_val
    results['Variance'] = var_val

    if np.isclose(mean_val, 0, atol=1e-2):
        results['Mean Test'] = "Mean close to zero (indicates white noise)"
    else:
        results['Mean Test'] = "Mean not close to zero (not white noise)"

    return results

if __name__ == "__main__":
    if len(sys.argv)  < 3:
        print("Usage: {} <sym> <date>".format(sys.argv[0]))
        sys.exit(1)
    # target_date = '2024-09-5'
    back_days = 350
    target_date = sys.argv[2]
    sym = sys.argv[1]
    syms=[sym]
    start_date = add_days_to_date(target_date, -back_days)
    data = yf.download(syms, start=start_date, end=target_date)
    df = data['Close']
    # pdb.set_trace()
    print(df.index[-1])
    x = np.diff(np.log(df.values[-50:]),1)
    print("mean rtn: ", np.mean(x))
    x = x - np.mean(x)

# Check if the array is white noise
    results = check_white_noise(x)

    # Print results
    for key, value in results.items():
        print(f"{key}: {value}")
