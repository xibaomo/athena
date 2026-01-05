import pdb

import robin_stocks.robinhood as rh
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pandas as pd
import yfinance as yf
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from option_chain import *

DATE_FORMAT = "%Y-%m-%d"


def download_from_yfinance(ticker, interval='1h', period='730d'):
    df = yf.download(ticker, period=period, interval=interval)
    bars_per_day = 7
    print("History downloaded. Days: ", len(df) / bars_per_day)
    return df, bars_per_day


def download_from_robinhood(ticker, interval='hour', span='3month'):
    historical_data = rh.stocks.get_stock_historicals(ticker, interval=interval, span=span)
    # Convert the historical data into a DataFrame
    df = pd.DataFrame(historical_data)
    # pdb.set_trace()
    # Convert the 'begins_at' column to datetime
    df['begins_at'] = pd.to_datetime(df['begins_at'])

    # Rename columns for better readability
    df.rename(columns={
        'begins_at': 'Date',
        'open_price': 'Open',
        'close_price': 'Close',
        'high_price': 'High',
        'low_price': 'Low',
        'volume': 'Volume'
    }, inplace=True)

    # Convert price columns to numeric values
    price_columns = ['Open', 'Close', 'High', 'Low']
    df[price_columns] = df[price_columns].apply(pd.to_numeric)
    # pdb.set_trace()
    return df, 6  # BARS_PER_DAY


def count_trading_days(start_date=None, end_date=None, exchange='NYSE'):
    """
    Calculate the number of trading days between two dates for a given exchange.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        exchange (str): The exchange to consider (default is 'NYSE').

    Returns:
        int: The number of trading days between the two dates.
    """
    if start_date is None:
        today = datetime.today()
        start_date = today.strftime("%Y-%m-%d")
    # Get the exchange calendar (default is NYSE)
    if exchange.upper() == 'NYSE':
        cal = mcal.get_calendar('NYSE')
    elif exchange.upper() == 'NASDAQ':
        cal = mcal.get_calendar('NASDAQ')
    # Add other exchanges as needed

    # Get the valid trading days between start and end date
    trading_days = cal.valid_days(start_date=start_date, end_date=end_date)

    # Return the number of trading days
    return len(trading_days)


class TradeDaysCounter(object):
    def __init__(self, exchange='NYSE'):
        if exchange.upper() == 'NYSE':
            self.cal = mcal.get_calendar('NYSE')
        elif exchange.upper() == 'NASDAQ':
            self.cal = mcal.get_calendar('NASDAQ')

    def countTradeDays(self, end_date, start_date=None):
        if start_date is None:
            today = datetime.today()
            start_date = today.strftime("%Y-%m-%d")
        trading_days = self.cal.valid_days(start_date=start_date, end_date=end_date)

        # Return the number of trading days
        return len(trading_days)


def natural_days_between_dates(start_date=None, end_date=None, date_format=DATE_FORMAT):
    """
    Calculate the number of days between two date strings.

    Args:
        start_date (str): The starting date as a string.
        end_date (str): The ending date as a string.
        date_format (str): The format in which the dates are provided. Default is "%Y-%m-%d".

    Returns:
        int: The number of days between the two dates.
    """
    if start_date is None:
        today = datetime.today()
        start_date = today.strftime("%Y-%m-%d")
    # Convert date strings to datetime objects
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)

    # Calculate the difference in days
    delta = end - start
    return delta.days


def eval_stability(rtns, n_intervals=10):
    '''
    estimate statbility of return series by computing averaging difference of CDFs
    '''
    spacing = len(rtns) // n_intervals
    print(f"stability check. spacing: {spacing}")
    ds = []
    for i in range(n_intervals - 1):
        r1 = rtns[i * spacing:(i + 1) * spacing]
        r2 = rtns[(i + 1) * spacing:(i + 2) * spacing]
        d, p = stats.ks_2samp(r1, r2)
        ds.append(d)

    return np.mean(ds)


def find_stablest_spacing(rtns, init_spacing, increment):
    spacing = init_spacing
    min_diff = 0
    best_spacing = init_spacing
    while 1:
        n_spacings = len(rtns) // spacing
        i0 = len(rtns) - n_spacings * spacing
        ds = []
        for i in range(n_spacings - 1):
            d, p = stats.ks_2samp(rtns[i0:i0 + spacing], rtns[i0 + spacing:i0 + 2 * spacing])
            i0 += spacing
            ds.append(p)
        # print(ds)
        aved = np.mean(ds)
        # print(len(ds),aved)
        if aved > min_diff:
            min_diff = aved
            best_spacing = spacing

        spacing += increment
        if len(rtns) // spacing < 2:
            break
    return best_spacing, min_diff


def compute_vol_price_log_slope(ticker, lookback):
    prd = str(lookback + 1) + "d"
    data = yf.download(ticker, period=prd, interval='1d')
    y = data['Volume'].values[-lookback:]
    x = np.linspace(0, len(y), len(y))
    p = np.polyfit(x, np.log(y), 1)
    # breakpoint()

    vls = p.flatten()[0]
    latest_ratio = (np.mean(y[-5:]) / np.mean(y)).flatten()[0]
    # breakpoint()
    print(f"\033[1;31m{lookback}-day vol log-slope: {vls:.4f}, last_week/mean(vol): {latest_ratio:.3f}\033[0m")

    y = data['Close'].values[-lookback:]
    x = np.linspace(0, len(y), len(y))
    p = np.polyfit(x, np.log(y), 1)
    # breakpoint()

    vls = p.flatten()[0]
    latest_ratio = (np.mean(y[-5:]) / np.mean(y)).flatten()[0]
    # breakpoint()
    print(f"\033[1;31m{lookback}-day Close log-slope: {vls:.4f}, last_week/mean(Close): {latest_ratio:.3f}\033[0m")

    return vls


def prepare_options(sym, exp_date):
    options = get_option_chain_alpha_vantage(sym)
    print(f"{len(options)} options downloaded")
    puts = []
    calls = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'put':
            puts.append(opt)
        if opt['expiration'] == exp_date and opt['type'] == 'call':
            calls.append(opt)

    print(f"{len(puts)} calls/puts returned")
    puts = sorted(puts, key=lambda x: float(x["strike"]))
    calls = sorted(calls, key=lambda x: float(x["strike"]))
    return calls, puts


def create_premium_cal(options, p0, type):
    strikes = []
    asks = []
    bids = []
    for optn in options:
        strikes.append(float(optn['strike']))
        asks.append(float(optn['ask']))
        bids.append(float(optn['bid']))
    strikes = np.array(strikes)
    asks = np.array(asks)
    bids = np.array(bids)
    f_ask = interp1d(strikes, asks, kind='cubic', fill_value="extrapolate")
    f_bid = interp1d(strikes, bids, kind='cubic', fill_value="extrapolate")
    bounds = [np.min(strikes), np.max(strikes)]
    # ask_cal = lambda x: f_ask(x) if x <= bounds[1] else  99990.
    # ask_cal = lambda x: 0 if x <= bounds[0] elif x>=bounds[1] x-p0 else f_ask(x)
    if type == "put":
        ask_cal = lambda x: 0 if x <= bounds[0] else (x - p0 if x >= bounds[1] else f_ask(x))
        bid_cal = lambda x: 0 if x <= bounds[0] else (x - p0 if x >= bounds[1] else f_bid(x))
    if type == "call":
        ask_cal = lambda x: 0 if x >= bounds[1] else (p0 - x if x <= bounds[0] else f_ask(x))
        bid_cal = lambda x: 0 if x >= bounds[1] else (p0 - x if x <= bounds[0] else f_bid(x))

    return ask_cal, bid_cal, bounds


def compute_call_put_parity_strike(p0, calls, puts):
    call_ask, _, bounds = create_premium_cal(calls, p0, "call")
    put_ask, _, _ = create_premium_cal(puts, p0, "put")

    a = bounds[0]
    b = bounds[1]
    while b - a > 1e-2:
        mid = (a + b) * .5
        cost = call_ask(mid) - put_ask(mid)
        if cost > 0:
            a = mid
        else:
            b = mid

    mid = (a + b) * .5
    return mid


def golden_section_search(f, a, b, tol=1e-6, max_iter=200):
    gr = (np.sqrt(5) - 1) / 2  # golden ratio ≈ 0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)

    xmin = (b + a) / 2
    return xmin, f(xmin)


def eval_calls_value(price, calls):
    val = 0.
    for call in calls:
        if call['strike'] < price:
            val += (price - call['strike']) * call['open_interest']
    return val


def eval_puts_value(price, puts):
    val = 0.
    for put in puts:
        if put['strike'] > price:
            val += (put['strike'] - price) * put['open_interest']
    return val


def eval_option_total_value(price, calls, puts):
    return eval_calls_value(price, calls) + eval_puts_value(price, puts)


def eval_max_pain(calls, puts):
    def find_strike_range(calls, puts):
        max_strike = 0
        min_strike = 999999
        for opt in calls:
            min_strike = min(min_strike, opt['strike'])
            max_strike = max(max_strike, opt['strike'])
        for opt in puts:
            min_strike = min(min_strike, opt['strike'])
            max_strike = max(max_strike, opt['strike'])
        return min_strike, max_strike

    def cost_func(price):
        # breakpoint()
        total_val = eval_calls_value(price, calls) + eval_puts_value(price, puts)
        return total_val

    a, b = find_strike_range(calls, puts)
    x, y = golden_section_search(cost_func, a, b)

    return x, y


import numpy as np
from scipy.stats import ks_2samp


def find_closest_cdf_subarray(x, y):
    """
    Find the subarray of y whose empirical CDF is closest to x.

    Parameters
    ----------
    x : array-like
        Reference array
    y : array-like
        Larger array to search

    Returns
    -------
    best_start : int
        Start index of best-matching subarray in y
    best_distance : float
        KS distance
    """

    x = np.asarray(x)
    y = np.asarray(y)

    n = len(x)
    m = len(y)

    if n > m:
        raise ValueError("Length of x must be <= length of y")

    best_distance = np.inf
    best_start = None

    for i in range(m - n + 1):
        y_sub = y[i:i + n]
        d = ks_2samp(x, y_sub).statistic

        if d < best_distance:
            best_distance = d
            best_start = i

    return best_start, best_distance


def best_y_length_via_cdf(x, n, L_min=5, L_max=None):
    """
    Find the best length L of y such that the CDF of z
    is closest to the CDF of zz.

    Parameters
    ----------
    x : array-like
        Time series (1D)
    n : int
        Length of z (last segment)
    L_min : int
        Minimum length of y to consider
    L_max : int or None
        Maximum length of y (defaults to feasible maximum)

    Returns
    -------
    best_L : int
        Optimal length of y
    best_distance : float
        KS distance between z and zz
    """

    x = np.asarray(x)
    T = len(x)

    if n >= T:
        raise ValueError("n must be smaller than the length of x")

    z = x[-n:]

    if L_max is None:
        # yy + zz + y + z must fit
        L_max = (T - 2 * n) // 2

    best_L = None
    best_distance = np.inf

    for L in range(L_min, L_max + 1):
        # y: immediately before z
        y_start = T - n - L
        y_end = T - n
        y = x[y_start:y_end]

        # search yy before y
        yy_search_end = y_start
        best_yy_start = None
        best_yy_dist = np.inf

        for i in range(0, yy_search_end - L + 1):
            yy = x[i:i + L]
            d = ks_2samp(yy, y).statistic

            if d < best_yy_dist:
                best_yy_dist = d
                best_yy_start = i

        if best_yy_start is None:
            continue

        # zz: immediately after yy
        zz_start = best_yy_start + L
        zz_end = zz_start + n

        if zz_end > T:
            continue

        zz = x[zz_start:zz_end]

        # compare z and zz
        d_final = ks_2samp(z, zz).statistic

        if d_final < best_distance:
            best_distance = d_final
            best_L = L

    return best_L, best_distance


import numpy as np
from scipy.stats import ks_2samp


def _evaluate_L(args):
    x, n, L = args
    T = len(x)

    z = x[-n:]

    # y immediately before z
    y_start = T - n - L
    y_end = T - n
    if y_start < 0:
        return None

    y = x[y_start:y_end]

    # search yy before y
    yy_search_end = y_start
    best_yy_start = None
    best_yy_dist = np.inf

    for i in range(0, yy_search_end - L + 1):
        yy = x[i:i + L]
        d = ks_2samp(yy, y).statistic
        if d < best_yy_dist:
            best_yy_dist = d
            best_yy_start = i

    if best_yy_start is None:
        return None

    # zz immediately after yy
    zz_start = best_yy_start + L
    zz_end = zz_start + n
    if zz_end > T:
        return None

    zz = x[zz_start:zz_end]

    d_final = ks_2samp(z, zz).statistic
    return (L, d_final)


from joblib import Parallel, delayed
import multiprocessing as mp


def best_y_length_via_cdf_parallel(
        x, n, L_min=5, L_max=None, n_jobs=None
):
    """
    Parallel version of best_y_length_via_cdf.
    """

    x = np.asarray(x)
    T = len(x)

    if n >= T:
        raise ValueError("n must be smaller than len(x)")

    if L_max is None:
        L_max = (T - 2 * n) // 2

    if n_jobs is None:
        n_jobs = mp.cpu_count()

    tasks = [(x, n, L) for L in range(L_min, L_max + 1)]

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size=1
    )(delayed(_evaluate_L)(t) for t in tasks)

    best_L = None
    best_dist = np.inf

    for r in results:
        if r is None:
            continue
        L, d = r
        if d < best_dist:
            best_dist = d
            best_L = L

    return best_L, best_dist


def corr_for_fixed_n(S, m, n, prefix):
    """
    Compute corr(x, y) for fixed n and given m > n.
    """
    T = len(S)
    K = T // (m + n)

    if K < 2:
        return np.nan

    x = np.empty(K)
    y = np.empty(K)

    for k in range(K):
        i = k * (m + n)
        x[k] = (prefix[i + m] - prefix[i]) / m
        y[k] = (prefix[i + m + n] - prefix[i + m]) / n

    return np.corrcoef(x, y)[0, 1]


def find_best_m_given_n(S, n, m_range, K_min=20):
    """
    Optimize m under constraint m > n.
    """
    S = np.asarray(S)
    T = len(S)

    prefix = np.zeros(T + 1)
    prefix[1:] = np.cumsum(S)

    best = {
        "corr": -np.inf,
        "m": None
    }

    for m in m_range:
        if m <= n:
            continue

        K = T // (m + n)
        if K < K_min:
            continue

        r = corr_for_fixed_n(S, m, n, prefix)
        if np.isnan(r):
            continue

        if r > best["corr"]:
            best.update(corr=r, m=m)

    return best


from scipy.stats import wasserstein_distance


def find_best_wasserstein_subarray(x, y, window_len):
    best_d = float("inf")
    best_i = None

    for i in range(len(y) - window_len + 1):
        sub = y[i:i + window_len]
        d = wasserstein_distance(x, sub)

        if d < best_d:
            best_d = d
            best_i = i

    return best_i, best_d


def extract_features(x):
    """
    Time-structure features
    """
    x = np.asarray(x)

    mean = x.mean()
    std = x.std()
    acf1 = np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0.0

    return np.array([mean, std, acf1])


def find_analogs(x,
        series,
        horizon,
        K=20,
        alpha=0.5,
):
    """
    Find K analogs of the most recent window.

    Returns:
        list of (start_idx, score)
    """
    series = np.asarray(series)

    x_recent = x
    lookback = len(x)
    feat_x = extract_features(x_recent)

    scores = []

    for i in range(0, len(series) - lookback - horizon):
        y = series[i:i + lookback]
        z_future = series[i + lookback:i + lookback + horizon]

        # skip NaN or bad windows
        if np.any(np.isnan(y)) or np.any(np.isnan(z_future)):
            continue

        # distribution distance
        d_dist = wasserstein_distance(x_recent, y)

        # structure distance
        feat_y = extract_features(y)
        d_struct = np.linalg.norm(feat_x - feat_y)

        # combined score
        score = alpha * d_dist + (1 - alpha) * d_struct

        scores.append((i, score))

    scores.sort(key=lambda x: x[1])

    return scores[:K]


def forecast_distribution(
        series,
        analogs,
        lookback,
        horizon
):
    """
    Merge empirical distributions from analog futures
    """
    future_samples = []

    for idx, _ in analogs:
        future = series[idx + lookback: idx + lookback + horizon]
        future_samples.append(future)

    return np.concatenate(future_samples)


def analog_distribution_forecast(x,
        series,
        horizon,
        K=20,
        alpha=0.5
):
    analogs = find_analogs(x,
        series=series,
        horizon=horizon,
        K=K,
        alpha=alpha
    )

    future_dist = forecast_distribution(
        series=series,
        analogs=analogs,
        lookback=len(x),
        horizon=horizon
    )

    return {
        "analogs": analogs,
        "future_samples": future_dist,
        "quantiles": np.quantile(future_dist, [0.05, 0.25, 0.5, 0.75, 0.95])
    }
