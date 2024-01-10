import os,sys
import pandas as pd
from port_conf import *
from datetime import datetime, timedelta
import yfinance as yf
import pdb

DATA_FILE = 'data_downloaded.csv'

def add_days_to_date(date, num_days):
    # Convert string to datetime object
    # pdb.set_trace()
    # date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the resulting date back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} port.yaml ")
        sys.exit(1)

    portconf = PortfolioConfig(sys.argv[1])
    target_date = portconf.getTargetDate()
    print("Target date: ", target_date)
    df = pd.read_csv(portconf.getSymFile(), comment='#')
    syms = df['<SYM>'].values
    NUM_SYMS = portconf.getNumSymbols()
    start_date = add_days_to_date(target_date, -portconf.getLookback())
    end_date = add_days_to_date(target_date, portconf.getLookforward())
    
    data = yf.download(syms.tolist(), start=start_date, end=end_date)['Close']
    data.to_csv(DATA_FILE,index=False)
    
    print("Downloaded data saved: ",DATA_FILE)

    