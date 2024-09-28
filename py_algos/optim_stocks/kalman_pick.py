import sys,os
import matplotlib.pyplot as plt
from download import DATA_FILE
from pick_opt import locate_target_date
from port_conf import PortfolioConfig
from kalman_motion import *
import pandas as pd
import numpy as np
import multiprocessing
import pdb
NUM_PROCS = multiprocessing.cpu_count()-2
class SymInfo(object):
    def __init__(self,sym_,x_,profit_):
        self.sym = sym_
        self.x   = x_
        self.profit = profit_
def cal_single_score(Z):
    x = calibrate_kalman_args(Z,100,0)
    profit,_ = cal_profit(x,Z,100)
    return profit

def cal_all_scores(df,tid,lookback):
    pool = multiprocessing.Pool(processes=NUM_PROCS)
    params = [df[sym].values[tid-lookback:tid+1] for sym in df_close.keys()]

    pfs = pool.map(cal_single_score,params)
    pfs = np.array(pfs)
    # pdb.set_trace()
    syms = df.keys().values
    sorted_id = np.argsort(pfs)
    return syms[sorted_id],pfs[sorted_id]

def verify_past_future(df,sym,global_tid,lookback,lookfwd, cap=10000):
    price = df[sym].values[global_tid-lookback:global_tid+1]
    Z = np.log(price)
    print("calibrating for ", sym)
    x = calibrate_kalman_args(Z,100,opt_method=1)
    dt,R,q=x
    xs,P = kalman2dmotion_adaptive(Z,q,dt)
    print("std: ", np.sqrt(P[0,0]),np.sqrt(P[1,1]),np.sqrt(P[2,2]))
    # pdb.set_trace()
    past_profit, _ = cal_profit(x, Z, 100)
    print("past std: ", np.sqrt(P[0, 0]), np.sqrt(P[1, 1]), np.sqrt(P[2, 2]))
    if xs[-1,1] < 0 or xs[-1,2] < 0 \
            or past_profit < 1000.:
        print("skipped. {},v: {:.2f},a: {:.2f},past profit: {:.2f} ".format(sym,xs[-1,1],xs[-1,2],past_profit))
        print("dt,R,q: ", dt,R,q)
        print("K: ", xs[-1,-1])
        return -1, -1, -1, -1, -1

    print("latest est. daily rtn: ", xs[-1,0]-xs[-2,0])
    print("est. v,a: ", xs[-1,1],xs[-1,2])

    price = df[sym].values[global_tid-lookback:global_tid+lookfwd]
    Z = np.log(price)
    dt,R,q = x
    # xs,P = kalman_motion(Z,R=R,q=q,dt=dt)
    xs,P = adaptive_kalman_motion(Z,q,dt)
    s = xs[:,0]
    v = xs[:,1]
    a = xs[:,2]

    p0 = -1
    c0 = cap
    trans=[]
    for i in range(lookback,len(v)):
        if v[i] > 0 and v[i]>v[i-1]  and p0 < 0:
            # pdb.set_trace()
            p0 = price[i]
            tr = [i-lookback,-1,-1]
            trans.append(tr)
        if v[i] < 0 and p0 > 0:
            # pdb.set_trace()
            print("deal closed: ", s[i]-s[i-1])
            cap = cap/p0*price[i]
            trans[-1][1] = i-lookback
            trans[-1][2] = price[i]/p0-1
            p0 = -1
            break

    if p0>0:
        cap = cap/p0*price[-1]
        trans[-1][1] = len(v)-1 - lookback - 1
        trans[-1][2] = price[-1] / p0 - 1

    # pdb.set_trace()
    print(trans)
    print(sym)
    print("dt,R,q,K: ",dt,R,q,xs[-1,-1])
    # fig, axs = plt.subplots(4, 1)
    # axs[0].plot(xs[lookback:,0],'.-')
    # axs[0].plot(Z[lookback:], 'r.-')
    # axs[1].plot(v[lookback:], '.-')
    # axs[1].axhline(y=0, color='red', linestyle='-')
    # axs[2].plot(xs[lookback:, -1], '.-')
    # axs[3].plot(xs[lookback:,2],'.-')
    # axs[3].axhline(y=0, color='red', linestyle='-')
    # print("past&future profits: ", past_profit,cap-c0)
    # print("var: ",P)
    # plt.show()
    return past_profit,dt,R,q,cap-c0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {sys.argv[0]} port.yaml <target_date>")
        sys.exit(1)

    portconf = PortfolioConfig(sys.argv[1])
    target_date = sys.argv[2]  # portconf.getTargetDate()
    data = pd.read_csv(DATA_FILE, comment='#', header=[0, 1], parse_dates=[0], index_col=0)
    data = data.dropna(axis=1)

    df_close = data['Close']
    global_tid = locate_target_date(target_date, df_close)
    if global_tid < 0:
        global_tid = len(df_close)-1
    lookback = portconf.getLookback()

    # syms,profits = cal_all_scores(df_close,global_tid,lookback)
    syms = df_close.keys()
    # syms = ['PGR']
    headers = ['sym','dt','R','q','past_profit','future_profit']
    df_res = pd.DataFrame(columns=headers)

    for sym in syms:
        past_pf,dt,R,q,future_pf = verify_past_future(df_close,sym,global_tid,lookback,100)
        if dt > 0:
            df_res.loc[len(df_res)] = [sym,dt,R,q,past_pf,future_pf]
            print("sym, past_profit, dt,R,q, future_profit: ", sym, past_pf,dt,R,q,future_pf)

    df_res.to_csv("df_res.csv",index=False)