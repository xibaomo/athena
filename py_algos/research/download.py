import os, sys
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pdb

DATA_FILE = 'data_downloaded.csv'

import yfinance as yf
import pandas as pd
import time

import yfinance as yf
import pandas as pd
import time

# Attempt to import the specific yfinance exception for precise error handling
try:
    from yfinance.exceptions import YFPricesMissingError
except ImportError:
    # Fallback class if the specific import path is inaccessible
    class YFPricesMissingError(Exception):
        pass


def download_two_pass(tickers, period="100d",interval="1d"):
    """
    Downloads tickers in a two-pass strategy: bulk first, then individual downloads
    for any failed tickers, merging the results into one DataFrame.
    """

    # --- Pass 1: Attempt Bulk Download ---
    print("--- PASS 1: Attempting Bulk Download ---")

    # yfinance returns a tuple of (successful_tickers, failed_tickers) when bulk fails
    # We use a large try/except block just in case yf.download() throws a non-bulk-related error

    bulk_data = yf.download(
        tickers,
       period=period,
        interval=interval,
        # progress=False,
        # Pass all download parameters here
    )

    df = bulk_data['Close']
    failed_tickers = []
    for ticker in df.keys():
        if ticker.endswith(' '):
            failed_tickers.append(ticker.strip())
            bulk_data = bulk_data.drop(columns=ticker,level=1)

    if len(failed_tickers) == 0:
        return bulk_data

    final_bulk_df = bulk_data

    # --- Pass 2: Individual Download for Failed Tickers ---

    individual_dfs = []
    if not final_bulk_df.empty:
        individual_dfs.append(final_bulk_df)
    if failed_tickers:
        print("\n--- PASS 2: Downloading remaining tickers individually ---")

        individual_success = {}
        remaining_tickers = list(failed_tickers)

        for ticker in remaining_tickers:
            print(f"--- Retrying {ticker}...")
            # 1. Individual download request
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                # progress=False
            )

            if data.empty:
                # raise YFPricesMissingError(f"No data returned for {ticker} in retry.")
                print(f"failed again to download {ticker}")
                continue

            # data_multi = pd.concat({ticker: data}, axis=1)
            # data_multi = data_multi.swaplevel(axis=1)
            # if ticker=='ZS':
            #     breakpoint()

            individual_dfs.append(data)
            print(f"Successfully retrieved {ticker}.")
            time.sleep(0.5)  # Pause briefly

        if individual_dfs:
            # pd.concat with axis=1 merges the MultiIndex columns side-by-side by aligning the Date index.
            final_merged_df = pd.concat(individual_dfs, axis=1)

            return final_merged_df




def add_days_to_date(date_str, num_days):
    # Convert string to datetime object
    # pdb.set_trace()
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the resulting date back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <lookback_days> [sym.csv] ")
        sys.exit(1)

    target_date = sys.argv[1]

    # df = pd.read_csv('sp505.csv', comment='#')
    # syms = df['<SYM>'].values

    # syms = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    syms = []
    if len(sys.argv) > 2:
        df = pd.read_csv(sys.argv[2], comment='#')
        private_syms = df['<SYM>'].values
        for s in private_syms:
            if not s in syms:
                syms.append(s)
                print("sym added: " , s)
    print("total syms: ", len(syms))
    lookback = int(sys.argv[1])

    prd = str(lookback)+"d"
    # data = yf.download(syms,period=prd,interval='1d')

    data = download_two_pass(syms,period=prd)
    print(data)
    if len(data) == 0:
        print("ERROR: download fails")
        sys.exit(1)
    data.to_csv(DATA_FILE, index=True)

    print("Downloaded data saved: ",DATA_FILE)


