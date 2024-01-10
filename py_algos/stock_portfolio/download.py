import os,sys
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
        print("Usage: {sys.argv[0]} <date> <lookback_days> ")
        sys.exit(1)

    target_date = sys.argv[1]

    df = pd.read_csv('sp505.csv', comment='#')
    syms = df['<SYM>'].values

    lookback = int(sys.argv[2])
    start_date = add_days_to_date(target_date, -lookback)
    end_date = target_date
    print("history range: ", start_date, end_date)
    data = yf.download(syms.tolist(), start=start_date, end=end_date)['Close']

    print(data)
    if len(data) == 0:
        print("ERROR: download fails")
        sys.exit(1)
    data.to_csv(DATA_FILE,index=True)
    
    print("Downloaded data saved: ",DATA_FILE)

    