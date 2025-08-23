import os, sys
import pandas as pd
from port_conf import *
from datetime import datetime, timedelta
import yfinance as yf
import pdb

DATA_FILE = 'data_downloaded.csv'

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
        print(f"Usage: {sys.argv[0]} <date> <lookback_days> [sym.csv] ")
        sys.exit(1)

    target_date = sys.argv[1]

    # df = pd.read_csv('sp505.csv', comment='#')
    # syms = df['<SYM>'].values

    syms = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    if len(sys.argv) > 3:
        df = pd.read_csv(sys.argv[3], comment='#')
        private_syms = df['<SYM>'].values
        for s in private_syms:
            if not s in syms:
                syms.append(s)
                print("sym added: " , s)
    print("total syms: ", len(syms))
    lookback = int(sys.argv[2])
    start_date = add_days_to_date(target_date, -lookback)
    end_date = target_date
    # print("history range: ", start_date, end_date)
    # data = yf.download(syms, start=start_date, end=end_date)
    data = yf.download(syms,period='730d',interval='1d')

    print(data)
    if len(data) == 0:
        print("ERROR: download fails")
        sys.exit(1)
    data.to_csv(DATA_FILE, index=True)

    print("Downloaded data saved: ",DATA_FILE)


