import pdb

import robin_stocks.robinhood as rh
from datetime import datetime,timedelta
import pandas_market_calendars as mcal
import pandas as pd
import yfinance as yf
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from option_chain import *
DATE_FORMAT ="%Y-%m-%d"

def download_from_yfinance(ticker,interval='1h',period='730d'):
    df = yf.download(ticker, period=period, interval=interval)
    bars_per_day=7
    print("History downloaded. Days: ",len(df)/bars_per_day)
    return df,bars_per_day

def download_from_robinhood(ticker,interval='hour',span='3month'):
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
    return df,6 #BARS_PER_DAY

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
    def __init__(self,exchange='NYSE'):
        if exchange.upper() == 'NYSE':
            self.cal = mcal.get_calendar('NYSE')
        elif exchange.upper() == 'NASDAQ':
            self.cal = mcal.get_calendar('NASDAQ')
    def countTradeDays(self,end_date,start_date=None):
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
    spacing = len(rtns)//n_intervals
    print(f"stability check. spacing: {spacing}")
    ds = []
    for i in range(n_intervals-1):
        r1 = rtns[i*spacing:(i+1)*spacing]
        r2 = rtns[(i+1)*spacing:(i+2)*spacing]
        d,p = stats.ks_2samp(r1,r2)
        ds.append(d)

    return np.mean(ds)
def find_stablest_spacing(rtns, init_spacing,increment):
    spacing = init_spacing
    min_diff = 0
    best_spacing = init_spacing
    while 1:
        n_spacings = len(rtns) // spacing
        i0 = len(rtns) - n_spacings * spacing
        ds = []
        for i in range(n_spacings-1):
            d,p = stats.ks_2samp(rtns[i0:i0+spacing],rtns[i0+spacing:i0+2*spacing])
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
    return best_spacing,min_diff

def compute_vol_price_log_slope(ticker,lookback):
    prd = str(lookback+1)+"d"
    data = yf.download(ticker, period=prd, interval='1d')
    y = data['Volume'].values[-lookback:]
    x = np.linspace(0,len(y),len(y))
    p = np.polyfit(x,np.log(y),1)
    # breakpoint()

    vls =  p.flatten()[0]
    latest_ratio = (np.mean(y[-5:])/np.mean(y)).flatten()[0]
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

def compute_call_put_parity_strike(p0,calls,puts):
    call_ask,_,bounds = create_premium_cal(calls,p0,"call")
    put_ask,_,_ = create_premium_cal(puts,p0,"put")

    a = bounds[0]
    b = bounds[1]
    while b-a > 1e-2:
        mid = (a+b)*.5
        cost = call_ask(mid) - put_ask(mid)
        if cost > 0:
            a = mid
        else:
            b = mid

    mid = (a+b)*.5
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
def eval_calls_value(price,calls):
    val = 0.
    for call in calls:
        if call['strike'] < price:
            val += (price - call['strike'])*call['open_interest']
    return val
def eval_puts_value(price,puts):
    val = 0.
    for put in puts:
        if put['strike'] > price:
            val += (put['strike'] - price)*put['open_interest']
    return val
def eval_option_total_value(price, calls,puts):
    return eval_calls_value(price, calls) + eval_puts_value(price,puts)
def eval_max_pain(calls,puts):
    def find_strike_range(calls,puts):
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
        total_val = eval_calls_value(price,calls) + eval_puts_value(price,puts)
        return total_val

    a,b = find_strike_range(calls,puts)
    x,y = golden_section_search(cost_func,a,b)

    return x,y





















