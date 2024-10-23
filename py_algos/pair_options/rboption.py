import pdb
import pickle,sys,os
import time
import random
from datetime import datetime,timedelta

import robin_stocks.robinhood as rh
from utils import *
from mkv_cal import *
import getpass


# Login to Robinhood
username = os.getenv("BROKER_USERNAME")
password = os.getenv("BROKER_PASSWD")
rh.login(username, password)
data_file='options.pkl'

ticker = sys.argv[1]
cur_price = float(rh.stocks.get_latest_price(ticker)[0])
# Function to get the option chain for a given stock ticker
def get_option_chain(ticker,target_date=None):
    if target_date is None:
        today = datetime.today()
        target_date = today + timedelta(days=15)
    print("Earliest expiration date: ", target_date.strftime("%Y-%m-%d"))
    # Get option instruments for the given ticker
    option_instruments = rh.options.find_tradable_options(ticker)

    if not option_instruments:
        print(f"No options available for {ticker}")
        return

    # Display the option chain data
    ncalls = 0
    nputs = 0
    option_market_data=[]
    for option in option_instruments:
        exp_date =  datetime.strptime(option['expiration_date'], "%Y-%m-%d")
        if exp_date < target_date:
            option['ask_price'] = -1
            continue
        for i in range(1): #try once
            option_market_data = rh.options.get_option_market_data_by_id(option['id'])
            # if len(option_market_data) > 0:
            #     break
            # time.sleep(random.random())
        if len(option_market_data)==0:
            option['ask_price'] = -1
        else:
            option['ask_price'] = float(option_market_data[0]['ask_price'])
        print(f"Latest: {cur_price}, Strike: {option['strike_price']}, {option['expiration_date']}, {option['type']}, ask: {option['ask_price']}")
        if option['type'] == 'call':
            ncalls+=1
        if option['type'] == 'put':
            nputs+=1
        # print(f"Type: {option['type']}, Premium: {option['adjusted_mark_price']}")
        # print(f"Implied Volatility: {option['implied_volatility']}\n")
    print("calls: {}, puts: {}".format(ncalls,nputs))
    return option_instruments

# Example usage: Get option chain for a stock ticker (e.g., 'AAPL')
# Log into Robinhood using the credentials

data_file = ticker + "_" + data_file

# options=get_option_chain(ticker)
# with open(data_file, 'wb') as f:
#     pickle.dump(options,f)

with open(data_file,'rb') as f:
    options = pickle.load(f)

call_opts=[]
put_opts=[]
for opt in options:
    if opt['ask_price'] < 0:
        continue
    if opt['type'] == 'call':
        call_opts.append(opt)
    if opt['type'] == 'put':
        put_opts.append(opt)

maxprof=-999999
call_profit=0
put_profit=0

r = 0.05
print("Latest price: ",cur_price)
max_put = -9999
min_call = 9999
mi,mj=-1,-1
for i in range(len(put_opts)): #maximize
    tmp = float(put_opts[i]['strike_price']) - 2 * float(put_opts[i]['ask_price'])
    if tmp > max_put:
        max_put = tmp 
        mj = i 
for i in range(len(call_opts)): # minimize
    tmp = float(call_opts[i]['strike_price']) + 2* float(call_opts[i]['ask_price'])
    if tmp < min_call:
        min_call = tmp 
        mi = i 

max_expect_prof = (2*r*cur_price + max_put - min_call)*.5
call_profit = (1+r)*cur_price - float(call_opts[mi]['strike_price']) - float(call_opts[mi]['ask_price']) - float(put_opts[mj]['ask_price'])
put_profit = float(put_opts[mj]['strike_price']) - (1-r)*cur_price - float(call_opts[mi]['ask_price']) - float(put_opts[mj]['ask_price'])
        
print(f"max expected profit: {max_expect_prof}, call: {call_profit}, put: {put_profit}")
print(call_opts[mi])
print(put_opts[mj])

date1 = call_opts[mi]['expiration_date']
date2 = put_opts[mj]['expiration_date']
days1 = count_trading_days(end_date=date1)
days2 = count_trading_days(end_date=date2)

days = days1 if days1 < days2 else days2
print("workdays to expire: ", days)

df = download_from_robinhood(ticker)

pbu,pbd = compProb1stHidBounds(ticker,df,days,ub_rtn=r,lb_rtn=-r)
print(f"1st-hit probability: {pbu:.3f},{pbd:.3f}, total: {pbu+pbd:.3f}")

# Log out after done
