import copy
import pdb

import pandas as pd
import numpy as np
import yfinance as yf
import os,sys
from ga_min import *
from port_conf import *
from datetime import datetime, timedelta
from scipy.optimize import minimize

SIGMA_INF = 1e6
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
        return SIGMA_INF
    ws = copy.deepcopy(wts)
    ws = np.append(ws,1-np.sum(ws))
    if np.any(ws < 0):
        return SIGMA_INF
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
        return SIGMA_INF
    ws = copy.deepcopy(wts)
    ws = np.append(ws,1-np.sum(ws))
    if np.any(ws < 0)  or np.any(ws > weight_bound):
        return SIGMA_INF
    s = 0.
    sds = sym_std
    for i in range(len(ws)):
        for j in range(i,len(ws)):
            if i == j:
                w = ws[i]
                sd = sds[i]
                s += w**2 * sd**2
            else:
                wi = ws[i]
                wj = ws[j]
                si = sds[i]
                sj = sds[j]
                s += 2*wi*wj*cm[i,j]*si*sj
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
def info_entropy(wts):
    s = 0.
    for p in wts:
        s+=-p*np.log(p)
    return s
def check_true_profit(data,global_tid,weights,capital):
    profits=[]
    start_price = data.iloc[global_tid,:]
    for tid in range(global_tid+1,len(data)):
        cur = data.iloc[tid,:]
        sym_rtn = cur / start_price - 1
        port_rtn = rtn_cost(weights,sym_rtn)
        profits.append(port_rtn*capital)

    print("\033[1m\033[91mProfits(low,high,final):${:.2f} ${:.2f} ${:.2f}\033[0m".format(min(profits),max(profits),profits[-1]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} port.yaml <date>")
        sys.exit(1)

    portconf = PortfolioConfig(sys.argv[1])
    gaconf = GAMinConfig(sys.argv[1])
    target_date = sys.argv[2]
    df = pd.read_csv(portconf.getSymFile(),comment='#')
    NUM_SYMS = portconf.getNumSymbols()
    start_date = add_days_to_date(target_date,-portconf.getLookback())
    end_date   = add_days_to_date(target_date,portconf.getLookforward())

    syms = portconf.getSymbols()
    official_syms = None
    if len(syms)==0:
        picked_rows = df.sample(n=NUM_SYMS)
        official_syms = picked_rows['<SYM>'].values
    else:
        official_syms = np.array(syms,dtype=object)
        print("Given sym list: ", syms)

    syms = official_syms + "=X"
    data = yf.download(syms.tolist(), start = start_date, end = end_date)['Close']
    official_syms = [s[:-2] for s in data.keys()]
    if len(data.keys()) < NUM_SYMS:
        print("Download is not complete. Try again later")
        sys.exit(1)
    global_tid = locate_target_date(target_date,data)
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
    tid = snap_target_date(target_date,monthly_avg_return_ma)
    sym_rtns = monthly_avg_return_ma.iloc[tid,:].values
    # pdb.set_trace()
    ### estimate std of return of each symbol
    monthly_avg_std = daily_rtns.resample('M').std()
    cm = daily_rtns.iloc[:global_tid+1,:].corr().values
    monthly_avg_std_ma = monthly_avg_std.rolling(window=portconf.getMAWindow()).mean()
    sym_std = monthly_avg_std_ma.iloc[tid, :].values

    # pdb.set_trace()
    weights = portconf.getSymWeights()
    cycles = 3 if len(weights) == 0 else 1

    muw = portconf.getMuWeight()
    sigma_bounds = portconf.getSigmaBounds()
    def obj_func(ws,cost_type):
        t1 = rtn_cost(ws,sym_rtns)
        t2 = std_cost(ws,cm,sym_std,weight_bound=portconf.getWeightBound())

        if t2 == SIGMA_INF or t2 < sigma_bounds[0] or t2 > sigma_bounds[1]:
            return 1e6
        if cost_type == 0:
            return (t2*1-t1*muw)*10000
        if cost_type == 1:
            # return -t1/t2
            return abs(t1)/t2-t2

    cost_type = portconf.getCostType()
    for gid in range(cycles):
        sol = None
        if len(weights) == 0:
            print("==================== Optimization starts ====================")
            print("sym_rtns: ",sym_rtns)
            # pdb.set_trace()
            sol,_ = ga_minimize(obj_func,cost_type,len(syms)-1,num_generations=gaconf.getNumGenerations(),population_size=gaconf.getPopulation(),
                                cross_prob=gaconf.getCrossProb(),mutation_rate=gaconf.getMutateProb())
            # Run the optimization using Nelder-Mead

            result = minimize(obj_func, sol, args=(cost_type), method='Nelder-Mead',options={'xtol': 1e-6})
            if abs(result.fun) < 100:
                sol = result.x
            print("Final cost: ",obj_func(sol,cost_type))

            #compute grad
            # grad = np.zeros(len(sol))
            # dx = 1e-6
            # for i in range(len(grad)):
            #     sol[i] += dx
            #     f1 = obj_func(sol,daily_rtns)
            #     sol[i] -= dx
            #     f2 = obj_func(sol, daily_rtns)
            #     grad[i] = (f1-f2)/dx
            # print("Derivative norm: ",np.linalg.norm(grad))

        else:
            sol = np.array(weights)[:-1]

        print('selected syms: ',official_syms)
        ss = np.append(sol,1-np.sum(sol))
        tmp = ",".join(["{:.3f}".format(element) for element in ss])
        print("weights: [{}]".format(tmp))
        print("entropy: ", info_entropy(ss))

        predicted_mu = rtn_cost(sol,sym_rtns)
        predicted_std = std_cost(sol,cm,sym_std)
        print("best mean of rtn: {:.6f}".format(predicted_mu))
        print("best std  of rtn: {:.6f}".format(predicted_std))

        durtn = len(data)-global_tid -1
        np.set_printoptions(precision=3)

        ub_rtn = (predicted_mu+predicted_std)*durtn
        lb_rtn = (predicted_mu-predicted_std)*durtn
        print("predicted monthly return ({} days): [{:.3f},{:.3f}]".format(durtn,lb_rtn,ub_rtn))

        print("********** Verification **********")
        invest = portconf.getCapitalAmount()
        print("predicted profit of ${}: [{:.2f}, {:.2f}]".format(invest,lb_rtn*invest,ub_rtn*invest))
        q4 = (ub_rtn-lb_rtn)*0.4 + lb_rtn
        q5 = (ub_rtn+lb_rtn)*0.5
        q6 = (ub_rtn-lb_rtn)*0.6 + lb_rtn
        print("predicted profit at [40%,50%,60%]: ${:.2f},${:.2f},${:.2f}".format(invest*q4,invest*q5,invest*q6))
        check_true_profit(data,global_tid,sol,invest)
        start_price = data.iloc[global_tid,:]
        end_price = data.iloc[-1,:]
        true_sym_rtns = (end_price/start_price-1.).values

        np.set_printoptions(precision=3)
        print("Estmt. monthly rtns: ", sym_rtns * durtn)
        print("Actual monthly rtns: ",true_sym_rtns)
        port_rtn = rtn_cost(sol,true_sym_rtns)
        # print("\033[1m\033[91mActual profit of ${}: {:.2f}\033[0m".format(invest,port_rtn*invest))
        print("profit quantile: {:.2f}".format((port_rtn-lb_rtn)/(ub_rtn-lb_rtn)))
