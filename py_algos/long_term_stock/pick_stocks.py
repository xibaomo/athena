import pandas as pd
import os,sys
import yfinance as yf
from datetime import datetime, timedelta
from alpha_vantage.fundamentaldata import FundamentalData
import pdb
API_KEY='ZHXCW0CK7QJFMHK0'
NUM_SYMS=50
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
        print("Usage: {sys.argv[0]} <date> <lookback_days> [sym.csv] ")
        sys.exit(1)

    target_date = sys.argv[1]

    # df = pd.read_csv('sp505.csv', comment='#')
    # syms = df['<SYM>'].values

    syms = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    syms.append('TSM')

    lookback = int(sys.argv[2])
    start_date = add_days_to_date(target_date, -lookback)
    end_date = target_date
    print("history range: ", start_date, end_date)
    data = yf.download(syms, start=start_date, end=end_date)['Close']

    daily_rtn = data.pct_change()

    mus = daily_rtn.mean()
    stds = daily_rtn.std()
    score = mus / stds
    sorted_score = score.sort_values(ascending=False)
    syms = sorted_score.index[:NUM_SYMS].tolist()

    df = pd.DataFrame()
    df.index = sorted_score.index[:NUM_SYMS]
    df['score'] = sorted_score[:NUM_SYMS]

    pe=[]
    pb=[]
    evt2rev=[]
    evt2prf=[]
    # for sym in syms:
    #     fd = FundamentalData(key=API_KEY, output_format='pandas')
    #     dff, _ = fd.get_company_overview(symbol=sym)
    #     pe.append(dff['TrailingPE'])
    #     pb.append(dff['PriceToBookRatio'])
    #     evt2rev.append(dff['EVToRevenue'])
    #     evt2prf.append(dff['EVToEBITDA'])
    #     # pdb.set_trace()
    #
    # df['PE'] = pe
    # df['PB'] = pb
    # df['EVToRevenue'] = evt2rev
    # df['EVToEBITDA'] = evt2prf

    df.to_csv('sp500ttm.csv',index=False)



