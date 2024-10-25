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

class FirstHitProbCal(object):
    def __init__(self,rtns,nstates,r):
        self.expireDate2probs = {}
        self.mkvcal = MkvAbsorbCal(nstates)
        self.mkvcal.buildTransMat(rtns,-r,r)
        d = 2*r/nstates
        self.mid = int(r/d)
    def get1stHitProbs(self,expire_date):
        if expire_date in self.expireDate2probs:
            return self.expireDate2probs[expire_date]

        steps = count_trading_days(end_date=expire_date)
        pbu, pbd = self.mkvcal.comp1stHitProb(steps, self.mid, -2, -1)
        res = [pbu,pbd]
        self.expireDate2probs[expire_date] = res
        return res

def computeBestPair(cur_price, rb, ticker, lookback, call_opts,put_opts):
    df = download_from_robinhood(ticker)
    rtns = df['Close'].pct_change().values[-lookback:]
    mi = -1
    mj = -1
    max_prof = -999999
    probcal = FirstHitProbCal(rtns,500,rb)
    # cur_price = float(rh.stocks.get_latest_price(ticker)[0])
    print("Latest price: ", cur_price)
    call_profit = -1
    put_profit = -1
    for i in range(len(call_opts)):
        callopt = call_opts[i]
        # pu = probcal.get1stHitProbs(callopt['expiration_date'])[0]
        callcost = float(callopt['ask_price'])
        call_prof = (1+rb)*cur_price - float(callopt['strike_price']) - float(callopt['ask_price'])
        call_days = callopt['days']
        for j in range(len(put_opts)):
            putopt = put_opts[j]
            put_days = putopt['days']
            expire_date = callopt['expiration_date'] if call_days < put_days else putopt['expiration_date']
            pu,pd = probcal.get1stHitProbs(expire_date)
            tmp_up = call_prof - float(putopt['ask_price'])
            tmp_dw = float(putopt['strike_price']) - cur_price*(1-rb) - float(putopt['ask_price']) - callcost

            expect_prof = tmp_up * pu + tmp_dw * pd
            if expect_prof > max_prof and pu+pd>=.7 and tmp_up*tmp_dw>0:
                max_prof = expect_prof
                mi = i
                mj = j
                # print("max expected profit: ", max_prof)
                call_profit = tmp_up
                put_profit = tmp_dw

    print(f"\033[91moptimal expected profit: {max_prof:.2f}, {call_profit:.2f}, {put_profit:.2f}\033[0m")
    # pdb.set_trace()
    days1 = call_opts[mi]['days']
    days2 = put_opts[mj]['days']
    expire_date = call_opts[mi]['expiration_date'] if days1 < days2 else put_opts[mj]['expiration_date']
    print("trade days: ", days1 if days1 < days2 else days2)
    pu,pd = probcal.get1stHitProbs(expire_date)
    print(f"up,down probs: {pu:.3f}, {pd:.3f}, {pu+pd:.3f}")

    print(call_opts[mi])
    print(put_opts[mj])
    return mi,mj

# Example usage: Get option chain for a stock ticker (e.g., 'AAPL')
# Log into Robinhood using the credentials

data_file = ticker + "_" + data_file

options=get_option_chain(ticker)
with open(data_file, 'wb') as f:
    pickle.dump(options,f)

with open(data_file,'rb') as f:
    options = pickle.load(f)

call_opts=[]
put_opts=[]
days_counter = TradeDaysCounter()
for opt in options:
    days = days_counter.countTradeDays(opt['expiration_date'])
    opt['days'] = days
    if opt['ask_price'] < 0:
        continue
    if opt['type'] == 'call':
        call_opts.append(opt)
    if opt['type'] == 'put':
        put_opts.append(opt)

print("call put options are ready")
maxprof=-999999
call_profit=0
put_profit=0

r = 0.02
for i in range(8):
    rb = 0.02*i +0.02
    print("r = ",rb)
    mi,mj = computeBestPair(cur_price, rb,ticker,70,call_opts,put_opts)

# Log out after done
