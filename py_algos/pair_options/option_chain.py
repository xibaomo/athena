import requests
import numpy as np
import pandas as pd
import random
from scipy.interpolate import interp1d
# API_KEYS=['135AZC6I9RH9MVFP']

API_KEYS = ['C5QIR42XY4V3ASG5',
            'VQBWJO6WCUJOMUIG',
            'A4L0CXXLQHSWW8ZS',
            'PCTFSKA5PM24BCO8',
            'FB149871PN4JIG2G',
            'AHFFPXFFDEZKCZD9',
            '1NWLPK6V8PAQ3AI7',
            'W33FXW2WPUVQL97O',
            '60NXDC2B867XO653',
            'EWPWZVNXREDO4UI4',
            'CX7YYU3BMBSM0ZLY',
            '0P6NEBPQEP3ESZVN',
            '6YSFQE4DTU5Q6M10',
            'OJCL6PKZQROYS0AW',
            'LKFWLB5NUCXCE0CK',
            '45OVBBNMRNB6MECZ',
            'O1WJLL2X75GSKIRJ',
            'R09FXDE84VPCF55D',
            '02S7TSM4DZH1U6TR',
            'Z5RUBBRKM3DNP2EU']

def random_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))
def get_option_chain_alpha_vantage(sym):
    idx = np.random.randint(len(API_KEYS))
    url = 'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=' + sym.upper() + "&apikey=" + API_KEYS[idx]
    headers = {"X-Forwarded-For": random_ip()}
    r = requests.get(url, headers=headers)
    data = r.json()
    if not 'data' in data.keys():
        print(data)
    options = data['data']

    return options

def delta2ask(options, delta=0.25):
    diffs = []
    for optn in options:
        diffs.append(abs(abs(float(optn['delta'])) - delta))

    closest_idx = np.argsort(diffs)[:4]
    deltas = []
    asks = []
    for idx in closest_idx:
        deltas.append(abs(float(options[idx]['delta'])))
        asks.append(float(options[idx]['ask']))

    f = interp1d(deltas, asks, kind='linear', fill_value='extrapolate')
    # breakpoint()
    res = f(delta)
    if res <= 0:
        return 0.01
    return res

def call_put_ask_ratio(delta,calls,puts):
    call_ask = delta2ask(calls, delta)
    put_ask = delta2ask(puts, delta)
    print(f"delta_.25 call ask: {call_ask:.2f}, put ask: {put_ask:.2f}")
    return call_ask/put_ask

def __prepare_callsputs(sym, exp_date):
    options = get_option_chain_alpha_vantage(sym)
    print(f"{len(options)} options downloaded")
    puts = []
    calls = []
    for opt in options:
        if opt['expiration'] == exp_date and opt['type'] == 'put':
            puts.append(opt)
        if opt['expiration'] == exp_date and opt['type'] == 'call':
            calls.append(opt)

    print(f"{len(puts)} calls returned")

    return calls,puts

def get_option_chain_market_data(sym,exp_date):
    url = "https://api.marketdata.app/v1/options/chain/"+sym.upper() + "/"
    params = {}
    params['expiration'] = exp_date
    token = 'WnJWbkdsVXpkVEwyUFNVQUhNQUhWN29Xczkxc19HYThPTnZuRHZmZy1rQT0'
    headers={
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(url, params=params,headers=headers)
    data = response.json()
    # breakpoint()
    symbols = data['optionSymbol']
    # quote_url = "https://api.marketdata.app/v1/options/quotes/"
    calls=[]
    puts = []
    print(f"{len(symbols)} options downloaded")
    print(f"expiration: {pd.to_datetime(data['expiration'][0],unit='s')}")
    for i in range(len(symbols)):
        opt={}
        opt['strike'] = data['strike'][i]
        opt['ask'] = data['ask'][i]
        opt['bid'] = data['bid'][i]
        opt['delta'] = data['delta'][i]
        if data['side'][i] == 'call':
            calls.append(opt)
        if data['side'][i] == 'put':
            puts.append(opt)
    return calls,puts
def prepare_callsputs(sym,exp_date):
    calls,puts = get_option_chain_market_data(sym, exp_date)
    return calls,puts







