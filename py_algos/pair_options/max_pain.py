import pandas as pd
import requests
from io import StringIO
import os,sys

from utils import eval_max_pain,prepare_callsputs,eval_option_total_value
import robin_stocks.robinhood as rh

def download_spy_stock_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    syms = df['Symbol'].tolist()
    df.to_csv('sp500.csv', index=False)
    return syms

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]}  <expiration_date>")
        sys.exit(1)

    exp_date = sys.argv[1]
    all_syms = download_spy_stock_list()
    sys.exit(1)

    cur_prices = []
    cur_tv = []
    max_pain_prices = []
    max_pain_tv = []
    tickers = []
    # all_syms=['NVR']
    for ticker in all_syms:
        # breakpoint()
        calls,puts = prepare_callsputs(ticker,exp_date=exp_date)
        if len(calls) == 0 or len(puts) == 0:
            continue
        tickers.append(ticker)
        cur_price = float(rh.stocks.get_latest_price(ticker)[0])
        cur_total_value = eval_option_total_value(cur_price,calls,puts)
        cur_prices.append(cur_price)
        cur_tv.append(cur_total_value)
        mp,mp_tv = eval_max_pain(calls,puts)
        max_pain_prices.append(mp)
        max_pain_tv.append(mp_tv)
        print(f"current price: {cur_price:.2f}, current total_value: {cur_total_value:.2f}")
        print(f"max_pain: {mp:.2f}, max_pain_total_value: {mp_tv:.2f}, perf: {mp_tv/cur_total_value-1:.2f}")

    df = pd.DataFrame(tickers, columns=["sym"])
    df['cur_price'] = cur_prices
    df['cur_tv'] = cur_tv
    df['max_pain_price'] = max_pain_prices
    df['max_pain_tv'] = max_pain_tv
    df['perf'] = df['max_pain_tv']/df['cur_tv']-1
    df = df[df["cur_tv"] >= 1e5]
    df=df.reset_index(drop=True)
    df = df.sort_values(by='perf')

    df.to_csv('max_pain.csv')




