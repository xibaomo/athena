import copy
import pdb
import re
import pandas as pd
import numpy as np
import yfinance as yf
import os, sys
from ga_min import *
from port_conf import *
from download import add_days_to_date, DATA_FILE
from datetime import datetime, timedelta
from scipy.optimize import minimize
from mkv_absorb import *
from sym_selection import *
import pdb
SIGMA_INF = 1e60
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
def __select_sym_mu_std(daily_rtn, num_sym, option=0): #0: pick ave-mu leading ones
    mus = daily_rtn.mean()
    stds= daily_rtn.std()
    score = mus/stds
    thd = 0.2/250
    print("number of syms with ave rtn > {}: {}".format(thd, (mus.values>thd).sum()))
    sorted_score = score.sort_values(ascending=False)
    syms = sorted_score.index[:num_sym]
    return syms.tolist()

def select_syms_mu_std(daily_rtn, num_syms, options=0):
    #use two scores
    mus = daily_rtn.mean()
    stds= daily_rtn.std()
    long_score = mus/stds

    half_lookback = int(len(daily_rtn)/2)
    half_df = daily_rtn.iloc[-half_lookback:, :]
    short_score = half_df.mean()/half_df.std()

    score = long_score+short_score
    thd = 0.2 / 250
    print("number of syms with ave rtn > {}: {}".format(thd, (mus.values > thd).sum()))
    sorted_score = score.sort_values(ascending=False)
    df = sorted_score[:num_syms*2]
    df = df.sample(n=num_syms)
    syms = df.index[:num_syms]
    return syms.tolist()

def select_syms_mkv(daily_rtns, num_syms, lb_rtn, ub_rtn):
    score = []
    mkvcal = MkvAbsorbCal(100)
    for sym in daily_rtns.keys():
        rtns = daily_rtns[sym]
        pr, sp = mkvcal.compWinProb(rtns, lb_rtn, ub_rtn)
        s1 = pr/sp
        pr, sp = mkvcal.compWinProb(rtns[-30:], lb_rtn, ub_rtn)
        s2 = pr/sp
        score.append(s1+s2)
        print("{}: {}, {}".format(sym, s1, s2))
        # print("score: {}, {}".format(sym, score[-1]))
    score = np.array(score)
    # pdb.set_trace()
    sorted_id = np.argsort(score)[::-1]
    all_syms = daily_rtns.keys().values[sorted_id]
    # print(score[sorted_id])
    # print(all_syms)

    return all_syms[:num_syms].tolist()

def rtn_cost(ws, sym_rtns):
    s = 0.
    if len(ws) != len(sym_rtns):
        print("!!!!!!!!!!!Wrong weight!!!!!!!!!!")
        sys.exit(1)
    for w, v in zip(ws, sym_rtns):
        s += w * v
    return s
def std_cost(ws, cm, sym_std, weight_bound=0.8):
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

def check_true_profit(data, global_tid, weights, capital, port_mu, port_std, end_tid):
    profits = []
    start_price = data.iloc[global_tid, :]
    for tid in range(global_tid + 1, end_tid+1):
        cur = data.iloc[tid, :]
        sym_rtn = cur / start_price - 1
        port_rtn = rtn_cost(weights, sym_rtn)
        if np.isnan(port_rtn):
            pdb.set_trace()
        profits.append(port_rtn * capital)

    print(
        "\033[1m\033[91mProfits(low, high, final): {:^8.2f} {:^8.2f} {:^8.2f}. mu: {:.4e}, std: {:.4e}\033[0m". \
        format(min(profits), max(profits), profits[-1], port_mu, port_std))
def computeRiskShare(wts, cm, sym_std):
    sd0 = std_cost(wts, cm, sym_std)
    risk_share=np.zeros(len(wts))
    dx = 1.e-5

    for i in range(len(wts)):
        wts[i] = wts[i]+dx
        s = std_cost(wts, cm, sym_std)
        wts[i]-= dx
        t = (s - sd0)/dx*wts[i]
        if abs(t/sd0) > 1:
            pdb.set_trace()
        risk_share[i] = t/sd0
    return risk_share

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} port.yaml <target_date>")
        sys.exit(1)

    portconf = PortfolioConfig(sys.argv[1])
    gaconf = GAMinConfig(sys.argv[1])
    target_date = sys.argv[2] #portconf.getTargetDate()
    data = pd.read_csv(DATA_FILE, comment='#',parse_dates=[0],index_col=0)
    data = data.dropna(axis=1)
    # pdb.set_trace()
    NUM_SYMS = portconf.getNumSymbols()
    start_date = add_days_to_date(target_date, -portconf.getLookback())
    end_date = add_days_to_date(target_date, portconf.getLookforward())

    # pdb.set_trace()
    daily_rtn = data.pct_change()
    # daily_rtn = daily_rtn.drop(daily_rtn.index[0])
    global_tid = locate_target_date(target_date, data)

    if global_tid < 0:
        print("Failed to find target date")
        sys.exit(1)

    # syms = select_syms_mu_std(daily_rtn.iloc[:global_tid+1, :], NUM_SYMS)
    # syms = select_syms_mkv(daily_rtn.iloc[:global_tid+1, :], NUM_SYMS, -0.1, 0.1)
    syms = select_syms_corr_dist(data.iloc[:global_tid+1, :], NUM_SYMS, portconf.getShortTermWeight())
    daily_rtns = daily_rtn[syms]
    df = data[syms]
    print('number of selected syms: ', len(syms))
    print('selected syms: ', syms)

    ### estimate expectation of return of each symbol
    sym_rtns = daily_rtns.iloc[:global_tid + 1].mean().values
    # pdb.set_trace()
    ### estimate std of return of each symbol
    sym_std = daily_rtns.iloc[:global_tid + 1].std().values
    cm = daily_rtns.iloc[:global_tid + 1, :].corr().values

######################## optimization #############################
    def obj_func(wts, cost_type):
        # pdb.set_trace()
        ws = np.array(wts)
        for i in range(len(ws)):
            if ws[i] < 0:
                ws[i] = 0.
        ws = ws/ws.sum()

        if cost_type == 2:
            rs = computeRiskShare(ws, cm, sym_std);
            return np.std(rs)

    cost_type = 2
    #pdb.set_trace()
    for gid in range(1):
        sol = None
        if True:
            print("==================== Minimization starts ====================")
            # sol, _ = ga_minimize(obj_func, cost_type, daily_rtns.shape[1],
            #                      num_generations=gaconf.getNumGenerations(), population_size=gaconf.getPopulation(),
            #                      cross_prob=gaconf.getCrossProb(), mutation_rate=gaconf.getMutateProb())
            # Run the optimization using Nelder-Mead
            sol = np.ones(NUM_SYMS)
            sol = sol/sol.sum()
            result = minimize(obj_func, sol, args=(cost_type), method='Nelder-Mead', options={'xatol': 1e-4})
            if abs(result.fun) < 100:
                sol = result.x
            final_cost = obj_func(sol, cost_type)
            print("Final cost: ", final_cost)
            sol = np.array(sol)

        for i in range(len(sol)):
            if sol[i] < 0:
                sol[i] = 0
        sol = sol/sol.sum()
        sort_id = np.argsort(sol)
        sorted_sol = sol[sort_id]
        # pdb.set_trace()
        sort_syms = np.array(syms)[sort_id]

        tmp = ",".join(["{}".format(element) for element in sort_syms])
        print('sorted syms: [{}]'.format(tmp))
        tmp = ",".join(["{:.3f}".format(element) for element in sorted_sol])
        print("weights: [{}]".format(tmp))

        predicted_mu = rtn_cost(sol, sym_rtns)
        predicted_std = std_cost(sol, cm, sym_std)
        print("best mean of daily rtn of portfolio: {:.6e}".format(predicted_mu))
        print("best std  of daily rtn of portfolio: {:.6f}".format(predicted_std))

        print("********** Verification **********")
        invest = portconf.getCapitalAmount()

        # pdb.set_trace()
        print("Calculating the true profit on ", end_date)
        end_tid = locate_target_date(end_date, df)
        check_true_profit(df, global_tid, sol, invest, predicted_mu, predicted_std, end_tid)
        start_price = df.iloc[global_tid, :]

        end_price = df.iloc[end_tid, :]
        true_sym_rtns = (end_price / start_price - 1.).values
        # pdb.set_trace()

        np.set_printoptions(precision=3)
        durtn = end_tid - global_tid
        tmp = sym_rtns[sort_id] * durtn
        tmp = ", ".join(["{:^8.4f}".format(element) for element in tmp])
        print("Estmt. total rtns: ", tmp)
        tmp = ", ".join(["{:^8.4f}".format(element) for element in true_sym_rtns[sort_id]])
        print("Actual total rtns: ", tmp)

        risk_share = computeRiskShare(sol, cm, sym_std)
        tmp = ", ".join(["{:^8.3f}".format(element) for element in risk_share])
        print("Risk share: ",tmp)

        port_rtn = rtn_cost(sol, true_sym_rtns)
        # print("\033[1m\033[91mActual profit of ${}: {:.2f}\033[0m".format(invest, port_rtn*invest))
        print("Actual profit: {:.2f}".format(invest * port_rtn))
        print("Portfolio actual total return: {:.3f}".format(port_rtn))
        print("End of cycle")
