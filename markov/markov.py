import numpy as np
import pandas as pd
import yaml
import sys, os
sys.path.append(os.environ['ATHENA_HOME'] + '/py_basics')
from logger import *
from basics import *
from scipy.optimize import minimize,golden,minimize_scalar
import pdb
class MarkovConfig(object):
    def __init__(self,cf):
        self.yamlDict = yaml.load(open(cf))

    def getReturnBounds(self):
        return self.yamlDict['MARKOV']['RETURN_BOUNDS']

    def getOptAlgo(self):
        return self.yamlDict['MARKOV']['OPTIMIZATION']

    def getZeroStateType(self):
        return self.yamlDict['MARKOV']['ZERO_STATE_TYPE']

    def getLookback(self):
        return self.yamlDict['MARKOV']['LOOKBACK']

    def getPosProbThreshold(self):
        return self.yamlDict['MARKOV']['POS_PROB_THRESHOLD']

class MkvZeroStateOpenPriceOnly(object):
    def __init__(self,df,price):
        self.hi = df[HIGH_KEY].values / price - 1.
        self.lw = df[LOW_KEY].values / price - 1.

    def __labelMinbars(self,tid_s,tid_e,tp_return,sl_return):
        self.labels = []
        for tid in range(tid_s, tid_e):
            if self.hi[tid] >= 0 and self.lw[tid] <= 0:
                self.labels.append(State.ORIGIN)
            if self.hi[tid] > tp_return:
                self.labels.append(State.TP)

            elif self.lw[tid] < sl_return:
                self.labels.append(State.SL)
            else:
                pass
    def compWinProb(self,tid_s,tid_e,tp_return,sl_return,disp=False):
        self.__labelMinbars(tid_s,tid_e,tp_return,sl_return)
        n01 = count_subarr(self.labels, [State.ORIGIN, State.TP])
        n02 = count_subarr(self.labels, [State.ORIGIN, State.SL])

        total = n01 + n02

        # print("n01 = {}, n02 = {}".format(n01, n02))
        if total < 2:
            p01 = 0
        else:
            p01 = (n01 + 1) / (total + 2)

        if disp:
            print("tp_rtn = {}, n01 = {}, n02 = {}, tp prob. = {} ".format(tp_return, n01, n02, p01))

        return p01


def buy_label_minbars_zs0(df,tid_s, tid_e, # not included
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


def comp_win_prob_zs0(labels):
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

def comp_win_prob_buy_zs0(x,price,df,tid_s,tid_e,disp=False):
    if len(x) == 2:
        tp = x[0]
        sl = x[1]
    if len(x) == 1:
        tp = x
        sl = -x

    labels = buy_label_minbars_zs0(df,tid_s,tid_e,price,tp,sl)
    # pdb.set_trace()
    wp,n01,n02 = comp_win_prob_zs0(labels)

    if disp:
        print("tp_rtn = {}, n01 = {}, n02 = {}, tp prob. = {} ".format(x, n01, n02,wp))
    return wp

def comp_profit_expectation(x, mkvcal, price, tid_s,tid_e,disp=False):
    tp = x
    sl = -x

    wp = mkvcal.compWinProb(tid_s,tid_e,tp,sl, disp)
    # ep = wp*price*x - (1-wp)*price*x
    # ep = wp*price*tp
    ep = wp
    return -ep

def max_prob_buy(zs,price,df,hist_start,hist_end,
                 bnds,algo):
    mkvcal = None
    if zs == 0:
        mkvcal = MkvZeroStateOpenPriceOnly(df,price)

    opt_func = comp_profit_expectation

    if algo == 0: # Powell
        Log(LOG_INFO) << "Powell optimization used"
        x0 = np.mean(bnds)
        res = minimize(opt_func, x0, (mkvcal, price, hist_start, hist_end), bounds=[bnds],
                       method="Powell", options={'xtol': 1e-3, 'disp': True, 'ftol': 1e-3})
        tp = res.x
    if algo == 1: # golden search
        Log(LOG_INFO) << "Golden search used"
        tp,_ = golden_search_min_prob(opt_func,args=(mkvcal, price, hist_start, hist_end),bounds=bnds)

    sl = -tp

    print("Optimized tp&sl: ", tp)
    # wp = comp_win_prob_buy_zs0(res.x,price,df,hist_start,hist_end,True)

    wp = mkvcal.compWinProb(hist_start,hist_end,tp,sl,True)
    return tp,wp
if __name__ == "__main__":
    Log.setlogLevel(LOG_INFO)
    if len(sys.argv) < 3:
        print("Usage: python {} <csv_file> <markov.yaml>".format(sys.argv[0]))
        sys.exit()

    csv_file = sys.argv[1]
    yml_file = sys.argv[2]
    df = pd.read_csv(csv_file,sep='\t')
    mkvconf = MarkovConfig(yml_file)

    hist_len = mkvconf.getLookback()
    for i in range(84321-2,len(df)-1):
        tarid = i
        hist_start = tarid - hist_len
        hist_end = tarid
        price = df[OPEN_KEY][tarid]

        x,pv = max_prob_buy(0,price,df,hist_start,hist_end,
                            mkvconf.getReturnBounds(),
                            mkvconf.getOptAlgo())

        print("best param: ",x)
        print("best prob.: ", pv)

