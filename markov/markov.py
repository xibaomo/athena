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

    def getOptTarget(self):
        return self.yamlDict['MARKOV']['OPT_TARGET']

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
    def getStartCount(self):
        n10 = count_subarr(self.labels,[State.TP,State.ORIGIN])
        n20 = count_subarr(self.labels,[State.SL,State.ORIGIN])
        return n10,n20

class MkvZeroStateEnds(object):
    def __init__(self,df,price):
        self.hi = df[HIGH_KEY].values / price - 1.
        self.lw = df[LOW_KEY].values / price - 1.

    def __labelMinbars(self,tid_s,tid_e,tp_return,sl_return):
        self.labels = []
        for tid in range(tid_s, tid_e):
            if self.hi[tid] <= tp_return and self.lw[tid] >= sl_return:
                self.labels.append(State.ORIGIN)
            elif self.hi[tid] > tp_return:
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

        p01 = 0
        if total < 1:
            p01 = 0
        else:
            p01 = (n01 + 1) / (total + 2)

        if disp:
            print("tp_rtn = {}, n01 = {}, n02 = {}, tp prob. = {} ".format(tp_return, n01, n02, p01))

        return p01

    def getStartCount(self):
        n10 = count_subarr(self.labels,[State.TP,State.ORIGIN])
        n20 = count_subarr(self.labels,[State.SL,State.ORIGIN])
        return n10,n20

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
    if total<10:
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

def comp_cost_func(x, mkvconf,mkvcal, price, tid_s,tid_e,disp=False):
    tp = x
    sl = -x

    wp = mkvcal.compWinProb(tid_s,tid_e,tp,sl, disp)

    if mkvconf.getOptTarget() == 0:
        ep = wp
    elif mkvconf.getOptTarget() == 1:
        ep = wp*price*tp
    elif mkvconf.getOptTarget() == 2:
        ep = wp * price * x - (1 - wp) * price * x
    else:
        pass
    return -ep

def max_prob_buy(mkvconf,price,df,hist_start,hist_end):
    mkvcal = None
    if mkvconf.getZeroStateType() == 0:
        mkvcal = MkvZeroStateOpenPriceOnly(df,price)
    elif mkvconf.getZeroStateType() == 1:
        mkvcal = MkvZeroStateEnds(df,price)
    else:
        pass

    opt_func = comp_cost_func

    if mkvconf.getOptAlgo() == 0: # Powell
        Log(LOG_INFO) << "Powell optimization used"
        x0 = np.mean(mkvconf.getReturnBounds())
        res = minimize(opt_func, x0, (mkvconf, mkvcal, price, hist_start, hist_end), bounds=[mkvconf.getReturnBounds()],
                       method="Powell", options={'xtol': 1e-3, 'disp': True, 'ftol': 1e-3})
        tp = res.x
    if mkvconf.getOptAlgo() == 1: # golden search
        Log(LOG_INFO) << "Golden search used"
        tp,_ = golden_search_min_prob(opt_func,args=(mkvconf,mkvcal, price, hist_start, hist_end),bounds=mkvconf.getReturnBounds())

    sl = -tp

    print("Optimized tp&sl: ", tp)
    # wp = comp_win_prob_buy_zs0(res.x,price,df,hist_start,hist_end,True)

    wp = mkvcal.compWinProb(hist_start,hist_end,tp,sl,True)

    if wp == 0:
        n10,n20 = mkvcal.getStartCount()
        print("n10 = {}, n20 = {}".format(n10,n20))
    return tp,wp,mkvcal
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

        x,pv = max_prob_buy(mkvconf,price,df,hist_start,hist_end)

        print("best param: ",x)
        print("best prob.: ", pv)
        if pv==0:
            print("price: ", price)
            break

