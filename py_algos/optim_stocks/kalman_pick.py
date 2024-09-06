import sys,os
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

    syms,profits = cal_all_scores(df_close,global_tid,lookback)