import statsmodels.api as sm

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_interval_quantiles(returns, spacing):
    """
    Split returns into intervals and compute quantiles for each interval.

    Parameters
    ----------
    returns : array-like
        Time series of returns.
    spacing : int
        Interval length (number of points per interval).
    quantiles : list
        List of quantiles to compute (e.g. [0.05, 0.5, 0.95]).

    Returns
    -------
    DataFrame: rows=intervals, cols=quantiles
    """

    quantiles = np.linspace(0.01,0.99,4)
    returns = np.asarray(returns)
    n = len(returns)
    n_intervals = n // spacing

    data = []
    for i in range(n_intervals):
        chunk = returns[i * spacing: (i + 1) * spacing]
        q_vals = np.quantile(chunk, quantiles)
        data.append(q_vals)

    return pd.DataFrame(data, columns=[f"q{int(q * 10000)}" for q in quantiles])


def predict_next_quantiles(quantile_df,lags=1):

    from statsmodels.tsa.api import VAR
    preds = {}
    model = VAR(quantile_df)
    fitted = model.fit(lags)
    print(fitted.summary())
    forecast = fitted.forecast(quantile_df.values[-lags:], steps=1)

    preds = dict(zip(quantile_df.columns, forecast[0]))
    breakpoint()
    return preds



# ---------------- Example ----------------
# np.random.seed(0)
# # simulate 200 daily returns
# returns = np.random.normal(0, 0.02, 200)
#
# # settings
# spacing = 50
# quantiles = [0.05, 0.5, 0.95]
#
# # compute quantiles per interval
# quantile_df = compute_interval_quantiles(returns, spacing, quantiles)
# print("Quantiles per interval:\n", quantile_df)
#
# # predict next interval quantiles
# preds = predict_next_quantiles(quantile_df)
# print("\nPredicted quantiles for next interval:\n", preds)
