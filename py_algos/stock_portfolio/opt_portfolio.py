import copy
import pdb
import re
import pandas as pd
import numpy as np
import yfinance as yf
import os, sys
from ga_min import *
from port_conf import *
from datetime import datetime, timedelta
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller

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


def __rtn_cost(wts, daily_rtns):
    ws = copy.deepcopy(wts)
    ws = np.append(ws, 1 - np.sum(ws))
    s = 0.
    monthly_avg_return = daily_rtns.resample('M').mean()
    # Compute 5-month moving average of monthly average daily return
    monthly_avg_return_ma = monthly_avg_return.rolling(window=5).mean()
    for w, v in zip(ws, monthly_avg_return_ma.iloc[-1, :].values):
        s += w * v
    return s


def rtn_cost(wts, sym_rtns):
    ws = copy.deepcopy(wts)
    ws = np.append(ws, 1 - np.sum(ws))
    s = 0.
    if len(ws) != len(sym_rtns):
        print("!!!!!!!!!!!Wrong weight!!!!!!!!!!")
        sys.exit(1)
    for w, v in zip(ws, sym_rtns):
        s += w * v
    return s


def __std_cost(wts, daily_rtns):
    if np.sum(wts) >= 1.:
        return SIGMA_INF
    ws = copy.deepcopy(wts)
    ws = np.append(ws, 1 - np.sum(ws))
    if np.any(ws < 0):
        return SIGMA_INF
    cm = daily_rtns.corr().values
    s = 0.
    monthly_avg_std = daily_rtns.resample('M').std()
    # Compute 5-month moving average of monthly average daily return
    monthly_avg_std_ma = monthly_avg_std.rolling(window=5).mean()
    sds = monthly_avg_std_ma.iloc[-1, :].values
    for i in range(len(ws)):
        for j in range(len(ws)):
            if i == j:
                w = ws[i]
                sd = sds[i]
                s += w ** 2 * sd ** 2
            else:
                wi = ws[i]
                wj = ws[j]
                si = sds[i]
                sj = sds[j]
                s += wi * wj * cm[i, j] * si * sj
    return np.sqrt(s)


def std_cost(wts, cm, sym_std, weight_bound=0.8):
    if np.sum(wts) >= 1.:
        return SIGMA_INF
    ws = copy.deepcopy(wts)
    ws = np.append(ws, 1 - np.sum(ws))
    if np.any(ws < 0) or np.any(ws > weight_bound):
        return SIGMA_INF
    s = 0.
    sds = sym_std
    for i in range(len(ws)):
        for j in range(i, len(ws)):
            if i == j:
                w = ws[i]
                sd = sds[i]
                s += w ** 2 * sd ** 2
            else:
                wi = ws[i]
                wj = ws[j]
                si = sds[i]
                sj = sds[j]
                s += 2 * wi * wj * cm[i, j] * si * sj
    return np.sqrt(s)


def snap_target_date(date_str, df):
    # pdb.set_trace()
    tar_date = pd.to_datetime(date_str)
    for i in range(len(df)):
        dt = tar_date - df.index[i]
        if dt.days < 3:
            return i


def locate_target_date(date_str, df):
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
        if p > 0:
            s += -p * np.log(p)
    return s


def check_true_profit(data, global_tid, weights, capital, port_mu, port_std, cost):
    profits = []
    start_price = data.iloc[global_tid, :]
    for tid in range(global_tid + 1, len(data)):
        cur = data.iloc[tid, :]
        sym_rtn = cur / start_price - 1
        port_rtn = rtn_cost(weights, sym_rtn)
        profits.append(port_rtn * capital)

    print(
        "\033[1m\033[91mProfits(low,high,final): {:^8.2f} {:^8.2f} {:^8.2f}. mu: {:.4e}, std: {:.4e}, cost: {:.4f}\033[0m". \
        format(min(profits), max(profits), profits[-1], port_mu, port_std, cost))


def check_stationarity(dayrtn, syms, tid):
    pv = []
    ss = syms.astype(str)
    for s in ss:
        r = dayrtn[s + '=X'].values[1:tid + 1]
        p = adfuller(r)[1]
        pv.append(p)
    tmp = ",".join(["{:.4f}".format(element) for element in pv])
    print("stationary p-val: ", tmp)


def remove_unstationary(dayrtn, tid):
    df = pd.DataFrame()
    # pdb.set_trace()
    for col in dayrtn.keys():
        r = dayrtn[col].values[1:tid + 1]
        pv = adfuller(r)[1]
        if pv < 0.01:
            df[col] = dayrtn[col]
    return df


def check_corr(dayrtn):
    cm = dayrtn.corr().values
    a = cm.ravel()
    a = np.sort(a)
    b = a[:-cm.shape[1]]
    print("ave corr: ", np.mean(b))
    # print(cm)


def pick_syms(dayrtn, num_syms):
    syms = []
    cm = dayrtn.corr().values
    a = cm.ravel()
    sort_id = np.argsort(a)
    cols = cm.shape[1]
    for i in range(len(sort_id)):
        row = int(sort_id[i] / cols)
        col = sort_id[i] % cols
        s1 = dayrtn.keys()[row]
        s2 = dayrtn.keys()[col]
        if not s1 in syms:
            syms.append(s1)
        if not s2 in syms:
            syms.append(s2)
        if len(syms) >= num_syms:
            break
    df = pd.DataFrame()
    syms = syms[:num_syms]
    for sym in syms:
        df[sym] = dayrtn[sym]
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} port.yaml <date>")
        sys.exit(1)

    portconf = PortfolioConfig(sys.argv[1])
    gaconf = GAMinConfig(sys.argv[1])
    target_date = sys.argv[2]
    df = pd.read_csv(portconf.getSymFile(), comment='#')
    NUM_SYMS = portconf.getNumSymbols()
    start_date = add_days_to_date(target_date, -portconf.getLookback())
    end_date = add_days_to_date(target_date, portconf.getLookforward())

    syms = portconf.getSymbols()
    official_syms = None
    if len(syms) == 0:
        official_syms = df['<SYM>'].values
    else:
        official_syms = np.array(syms, dtype=object)
        print("Given sym list: ", syms)

    syms = official_syms
    data = yf.download(syms.tolist(), start=start_date, end=end_date)['Close']
    daily_rtns = data.pct_change()
    global_tid = locate_target_date(target_date, data)
    # pdb.set_trace()
    daily_rtns = remove_unstationary(daily_rtns, global_tid)
    if len(daily_rtns.keys()) <= NUM_SYMS:
        print("only {} forexes have stationary history < {}, select all of them".format(daily_rtns.shape[1], NUM_SYMS))
        NUM_SYMS = daily_rtns.shape[1]
    else:
        daily_rtns = daily_rtns.sample(NUM_SYMS, axis=1)
        # daily_rtns = pick_syms(daily_rtns,NUM_SYMS)

    check_corr(daily_rtns)
    org_syms = [s[:-2] for s in daily_rtns.keys()]
    print("seleted syms: ", len(org_syms))
    print(org_syms)
    df = pd.DataFrame()  # keep track of original prices
    for k in daily_rtns.keys():
        df[k] = data[k]

    if global_tid < 0:
        print("Failed to find target date")
        sys.exit(1)
    print('Target date: ', data.index[global_tid])

    # pdb.set_trace()
    ### estimate expectation of return of each symbol
    sym_rtns = daily_rtns.iloc[:global_tid + 1].mean().values
    # pdb.set_trace()
    ### estimate std of return of each symbol
    sym_std = daily_rtns.iloc[:global_tid + 1].std().values
    print("past std: ", sym_std)
    print("ave past std: ", np.mean(sym_std))
    cm = daily_rtns.iloc[:global_tid + 1, :].corr().values
    # pdb.set_trace()
    weights = portconf.getSymWeights()
    cycles = 3 if len(weights) == 0 else 1

    muw = portconf.getMuWeight()
    sigma_bounds = portconf.getSigmaBounds()


    def obj_func(ws, cost_type):
        t1 = rtn_cost(ws, sym_rtns)
        t2 = std_cost(ws, cm, sym_std, weight_bound=portconf.getWeightBound())

        if t2 == SIGMA_INF or t2 < sigma_bounds[0] or t2 > sigma_bounds[1]:
            return 1e6
        if cost_type == 0:
            return (t2 * 1 - t1 * muw) * 10000
        if cost_type == 1:
            return -(t1 + 0e-3) / t2
            # return 10*abs(t1)-t2
            # return (abs(t1-0e-4)+1e-7)/t2/t2


    cost_type = portconf.getCostType()
    for gid in range(cycles):
        sol = None
        if len(weights) == 0:
            print("==================== Optimization starts ====================")
            # print("sym_rtns: ",sym_rtns)
            # pdb.set_trace()
            sol, _ = ga_minimize(obj_func, cost_type, daily_rtns.shape[1] - 1,
                                 num_generations=gaconf.getNumGenerations(), population_size=gaconf.getPopulation(),
                                 cross_prob=gaconf.getCrossProb(), mutation_rate=gaconf.getMutateProb())
            # Run the optimization using Nelder-Mead

            result = minimize(obj_func, sol, args=(cost_type), method='Nelder-Mead', options={'xtol': 1e-6})
            if abs(result.fun) < 100:
                sol = result.x
            final_cost = obj_func(sol, cost_type)
            print("Final cost: ", final_cost)

        else:
            sol = np.array(weights)[:-1]

        ss = np.append(sol, 1 - np.sum(sol))
        sort_id = np.argsort(ss)
        ss = ss[sort_id]
        # pdb.set_trace()
        sort_syms = np.array(org_syms)[sort_id]

        tmp = ",".join(["{}".format(element) for element in sort_syms])
        print('sorted syms: [{}]'.format(tmp))
        tmp = ",".join(["{:.3f}".format(element) for element in ss])
        print("weights: [{}]".format(tmp))
        print("entropy: ", info_entropy(ss))

        predicted_mu = rtn_cost(sol, sym_rtns)
        predicted_std = std_cost(sol, cm, sym_std)
        print("best mean of daily rtn of portfolio: {:.6e}".format(predicted_mu))
        print("best std  of daily rtn of portfolio: {:.6f}".format(predicted_std))

        durtn = len(data) - global_tid - 1
        np.set_printoptions(precision=3)

        ub_rtn = (predicted_mu + predicted_std) * durtn
        lb_rtn = (predicted_mu - predicted_std) * durtn
        print("predicted total return ({} days): [{:.3f},{:.3f}]".format(durtn, lb_rtn, ub_rtn))

        print("********** Verification **********")
        invest = portconf.getCapitalAmount()
        print("predicted profit of ${}: [{:.2f}, {:.2f}]".format(invest, lb_rtn * invest, ub_rtn * invest))

        check_true_profit(df, global_tid, sol, invest, predicted_mu, predicted_std, final_cost)
        start_price = df.iloc[global_tid, :]
        end_price = df.iloc[-1, :]
        true_sym_rtns = (end_price / start_price - 1.).values
        # pdb.set_trace()

        np.set_printoptions(precision=3)
        tmp = sym_rtns[sort_id] * durtn
        tmp = ", ".join(["{:^8.4f}".format(element) for element in tmp])
        print("Estmt. total rtns: ", tmp)
        tmp = ", ".join(["{:^8.4f}".format(element) for element in true_sym_rtns[sort_id]])
        print("Actual total rtns: ", tmp)

        port_rtn = rtn_cost(sol, true_sym_rtns)
        # print("\033[1m\033[91mActual profit of ${}: {:.2f}\033[0m".format(invest,port_rtn*invest))
        print("Actual profit: {:.2f}".format(invest * port_rtn))
        print("Portfolio actual total return: {:.3f}".format(port_rtn))
        print("End of cycle")
