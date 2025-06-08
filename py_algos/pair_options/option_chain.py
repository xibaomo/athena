import requests
import numpy as np
import random

API_KEYS = ['C5QIR42XY4V3ASG5',
            'VQBWJO6WCUJOMUIG',
            'A4L0CXXLQHSWW8ZS',
            'PCTFSKA5PM24BCO8',
            'FB149871PN4JIG2G',
            'AHFFPXFFDEZKCZD9',
            '1NWLPK6V8PAQ3AI7',
            'W33FXW2WPUVQL97O']

def random_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))
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
