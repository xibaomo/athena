import copy
import pdb

import pandas as pd
import numpy as np
import yfinance as yf
import os,sys
from ga_min import *
from port_conf import *
from datetime import datetime, timedelta

def add_days_to_date(date_str, num_days):
    # Convert string to datetime object
    # pdb.set_trace()
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the resulting date back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str

def __rtn_cost(wts,daily_rtns):
    ws = copy.deepcopy(wts)
    ws = np.append(ws,1-np.sum(ws))
    s = 0.
    monthly_avg_return = daily_rtns.resample('M').mean()
    # Compute 5-month moving average of monthly average daily return
    monthly_avg_return_ma = monthly_avg_return.rolling(window=5).mean()
    for w,v in zip(ws,monthly_avg_return_ma.iloc[-1,:].values):
        s += w*v
    return s
def rtn_cost(wts,sym_rtns):
    ws = copy.deepcopy(wts)
    ws = np.append(ws, 1 - np.sum(ws))
    s = 0.
    for w,v in zip(ws,sym_rtns):
        s += w*v
    return s
def __std_cost(wts,daily_rtns):
    if np.sum(wts) >= 1.:
        return 100
    ws = copy.deepcopy(wts)
    ws = np.append(ws,1-np.sum(ws))
    if np.any(ws < 0):
        return 100
    cm = daily_rtns.corr().values
    s = 0.
    monthly_avg_std = daily_rtns.resample('M').std()
    # Compute 5-month moving average of monthly average daily return
    monthly_avg_std_ma = monthly_avg_std.rolling(window=5).mean()
    sds = monthly_avg_std_ma.iloc[-1,:].values
    for i in range(len(ws)):
        for j in range(len(ws)):
            if i == j:
                w = ws[i]
                sd = sds[i]
                s += w**2 * sd**2
            else:
                wi = ws[i]
                wj = ws[j]
                si = sds[i]
                sj = sds[j]
                s += wi*wj*cm[i,j]*si*sj
    return np.sqrt(s)
def std_cost(wts,cm,sym_std,weight_bound=0.8):
    if np.sum(wts) >= 1.:
        return 100
    ws = copy.deepcopy(wts)
    ws = np.append(ws,1-np.sum(ws))
    if np.any(ws < 0)  or np.any(ws > weight_bound):
        return 100
    s = 0.
    sds = sym_std
    for i in range(len(ws)):
        for j in range(len(ws)):
            if i == j:
                w = ws[i]
                sd = sds[i]
                s += w**2 * sd**2
            else:
                wi = ws[i]
                wj = ws[j]
                si = sds[i]
                sj = sds[j]
                s += wi*wj*cm[i,j]*si*sj
    return np.sqrt(s)

def snap_target_date(date_str,df):
    # pdb.set_trace()
    tar_date = pd.to_datetime(date_str)
    for i in range(len(df)):
        dt = tar_date - df.index[i]
        if dt.days < 3:
            return i
def locate_target_date(date_str,df):
    # pdb.set_trace()
    tar_date = pd.to_datetime(date_str)
    prev_days = 1000000
    for i in range(len(df)):
        dt = tar_date - df.index[i]
        if dt.days == 0:
            return i
        if prev_days * dt.days < 0:
            return i
        prev_days = dt.days
    return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} port.yaml")
        sys.exit(1)

    portconf = PortfolioConfig(sys.argv[1])
    gaconf = GAMinConfig(sys.argv[1])
    df = pd.read_csv(portconf.getSymFile(),comment='#')
    NUM_SYMS = portconf.getNumSymbols()
    start_date = add_days_to_date(portconf.getTargetDate(),-portconf.getLookback())
    end_date   = add_days_to_date(portconf.getTargetDate(),portconf.getLookforward())

    picked_rows = df.sample(n=NUM_SYMS)
    syms = picked_rows['<SYM>'].values + "=X"

    data = yf.download(syms.tolist(), start = start_date, end = end_date)['Adj Close']
    if len(data.keys()) < NUM_SYMS:
        print("Download is not complete. Try again later")
        sys.exit(1)
    global_tid = locate_target_date(portconf.getTargetDate(),data)
    if global_tid < 0:
        print("Failed to find target date");
        sys.exit(1)
    print('Target date: ',data.index[global_tid])
    daily_rtns = data.pct_change()
    # pdb.set_trace()
    ### estimate expectation of return of each symbol
    monthly_avg_return = daily_rtns.resample('M').mean()
    monthly_avg_return_ma = monthly_avg_return.rolling(window=portconf.getMAWindow()).mean()
    # pdb.set_trace()
    tid = snap_target_date(portconf.getTargetDate(),monthly_avg_return_ma)
    sym_rtns = monthly_avg_return_ma.iloc[tid,:].values
    # pdb.set_trace()
    ### estimate std of return of each symbol
    monthly_avg_std = daily_rtns.resample('M').std()
    cm = daily_rtns.iloc[:global_tid+1,:].corr().values
    monthly_avg_std_ma = monthly_avg_std.rolling(window=portconf.getMAWindow()).mean()
    sym_std = monthly_avg_std_ma.iloc[tid, :].values
    def obj_func(ws,dayrtn):
        t1 = rtn_cost(ws,sym_rtns)
        t2 = std_cost(ws,cm,sym_std,weight_bound=portconf.getWeightBound())
        return (t2*1-t1*1)

    for gid in range(3):
        print("==================== Optimization starts ====================")
        sol,_ = ga_minimize(obj_func,daily_rtns,len(syms)-1,num_generations=gaconf.getNumGenerations(),population_size=gaconf.getPopulation(),
                            cross_prob=gaconf.getCrossProb(),mutation_rate=gaconf.getMutateProb())

        print("selected syms: ",picked_rows['<SYM>'].values)
        ss = np.append(sol,1-np.sum(sol))
        print(" ".join(["{:.3f}".format(element) for element in ss]))

        predicted_mu = rtn_cost(sol,sym_rtns)
        predicted_std = std_cost(sol,cm,sym_std)
        print("best mean of rtn: {:.6f}".format(predicted_mu))
        print("best std  of rtn: {:.6f}".format(predicted_std))

        durtn = len(data)-global_tid -1
        ub_rtn = (predicted_mu+predicted_std)*durtn
        lb_rtn = (predicted_mu-predicted_std)*durtn
        print("predicted monthly return ({} days): [{:.3f},{:.3f}]".format(durtn,lb_rtn,ub_rtn))

        print("********** Verfication **********")
        invest = portconf.getCapitalAmount()
        print("predicted profit of ${}: [{:.2f}, {:.2f}]".format(invest,lb_rtn*invest,ub_rtn*invest))
        start_price = data.iloc[global_tid,:]
        end_price = data.iloc[-1,:]
        sym_rtns = (end_price/start_price-1.).values
        port_rtn = rtn_cost(sol,sym_rtns)
        print("Actual profit of ${}: {:.2f}".format(invest,port_rtn*invest))
