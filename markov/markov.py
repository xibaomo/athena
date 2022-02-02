import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.environ['ATHENA_HOME'] + '/py_basics')
from logger import *
from basics import *
from scipy.optimize import minimize
import pdb

def buy_label_minbars(df,tid_s, tid_e, # not included
                  price,tp_return,sl_return):
    labels = np.empty(tid_e-tid_s)
    # op = df[OPEN_KEY].values / price - 1.
    hi = df[HIGH_KEY].values / price - 1.
    lw = df[LOW_KEY].values  / price - 1.
    k=0
    for tid in range(tid_s,tid_e):
        if hi[tid] >= tp_return and lw[tid] <= sl_return:
            b = State.NONE

        elif hi[tid] > tp_return:
            b = State.TP

        elif lw[tid] < sl_return:
            b = State.SL

        else:
            b = State.LIVE

        labels[k] = b
        k+=1

    return labels


def comp_win_prob(labels):

    n00 = count_subarr(labels,[State.LIVE,State.LIVE])
    n01 = count_subarr(labels,[State.LIVE,State.TP])
    # n02 = count_subarr(labels,[State.LIVE,State.SL])
    # n03 = count_subarr(labels,[State.LIVE,State.NONE])

    total = sum(labels==State.LIVE)

    # pdb.set_trace()
    p00 = n00/total
    p01 = n01/total

    print("n00 = {}, n01 = {}".format(n00,n01))

    wp = 1./(1-p00) * p01
    return wp

def comp_win_prob_buy(x,price,df,tid_s,tid_e):
    labels = buy_label_minbars(df,tid_s,tid_e,price,x,-x)
    wp = comp_win_prob(labels)
    print(x,wp)
    return -wp
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python {} <csv_file> <markov.yaml>".format(sys.argv[0]))
        sys.exit()

    csv_file = sys.argv[1]
    yml_file = sys.argv[2]
    df = pd.read_csv(csv_file,sep='\t')

    test_size = 100
    hist_len = 5000
    tarid = 69487
    hist_start = tarid - hist_len
    hist_end = tarid
    price = df[OPEN_KEY][tarid]

    # x0=[0.001,-0.001]
    x0=0.0011
    # bnds = ((0.001,0.005),(-0.01,0))
    bnds = ((0.001,None))
    res = minimize(comp_win_prob_buy,x0,(price,df,hist_start,hist_end),bounds=bnds,
                   method='Powell', options={'xtol': 1e-3, 'disp': True,'ftol':1e-2})
    # res = minimize(comp_win_prob_buy, x0, (price, df, hist_start, hist_end), bounds=bnds,
    #                method='Nelder-Mead', options={'disp': True, 'fatol': 1e-2})
    # labels = buy_label_minbars(df,0,hist_end,price,0.0033,-0.005)
    # wp = comp_win_prob(labels)
    # print("win prob: ",wp)
    print("best param: ",res.x)
