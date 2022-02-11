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
    labels=[]
    # op = df[OPEN_KEY].values / price - 1.
    hi = df[HIGH_KEY].values / price - 1.
    lw = df[LOW_KEY].values  / price - 1.
    k=0
    for tid in range(tid_s,tid_e):
        if hi[tid] >= 0 and lw[tid] <= 0:
            labels.append(State.ORIGIN)

        if hi[tid] > tp_return:
            labels.append(State.TP)

        elif lw[tid] < sl_return:
            labels.append(State.SL)

        else:
            pass

    return labels


def comp_win_prob(labels):
    # pdb.set_trace()
    # n00 = count_subarr(labels,[State.LIVE,State.LIVE])
    n01 = count_subarr(labels,[State.ORIGIN,State.TP])
    n02 = count_subarr(labels,[State.ORIGIN,State.SL])
    # n03 = count_subarr(labels,[State.LIVE,State.NONE])

    total =n01+n02

    # print("n01 = {}, n02 = {}".format(n01, n02))
    if total<2:
        p01 = 0
    else:
        p01 = (n01+1)/(total+2)
    # p02 = n02/total

    return p01,n01,n02

def comp_win_prob_buy(x,price,df,tid_s,tid_e,disp=False):
    if len(x) == 2:
        tp = x[0]
        sl = x[1]
    if len(x) == 1:
        tp = x
        sl = -x

    labels = buy_label_minbars(df,tid_s,tid_e,price,tp,sl)
    # pdb.set_trace()
    wp,n01,n02 = comp_win_prob(labels)

    if disp:
        print("n01 = {}, n02 = {}, tp prob. = {} ".format(n01, n02,wp))
    return -wp

def max_prob_buy(price,df,hist_start,hist_end,
                 bnds,algo):
    x0 = np.mean(bnds)

    res = minimize(comp_win_prob_buy, x0, (price, df, hist_start, hist_end), bounds=[bnds],
                   method=algo, options={'xtol': 1e-3, 'disp': True, 'ftol': 1e-2})

    print("Optimized tp&sl: ", res.x)
    comp_win_prob_buy(res.x,price,df,hist_start,hist_end,True)

    return res.x,-res.fun
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python {} <csv_file> <markov.yaml>".format(sys.argv[0]))
        sys.exit()

    csv_file = sys.argv[1]
    yml_file = sys.argv[2]
    df = pd.read_csv(csv_file,sep='\t')

    test_size = 100
    hist_len = 5000
    tarid = len(df)-1
    hist_start = tarid - hist_len
    hist_end = tarid
    price = df[OPEN_KEY][tarid]

    bnds = [0.002,0.004]
    x,pv = max_prob_buy(price,df,hist_start,hist_end,bnds,'Powell')
    # res = minimize(comp_win_prob_buy,x0,(price,df,hist_start,hist_end),bounds=bnds,
    #                method='Powell', options={'xtol': 1e-3, 'disp': True,'ftol':1e-2})
    # res = minimize(comp_win_prob_buy, x0, (price, df, hist_start, hist_end), bounds=bnds,
    #                method='Nelder-Mead', options={'disp': True, 'fatol': 1e-2})
    # labels = buy_label_minbars(df,0,hist_end,price,0.0033,-0.005)
    # wp = comp_win_prob(labels)
    # print("win prob: ",wp)
    print("best param: ",x)
    print("best prob.: ", pv)
