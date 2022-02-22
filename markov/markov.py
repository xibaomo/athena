import numpy as np
import pandas as pd
import yaml
import sys, os
sys.path.append(os.environ['ATHENA_HOME'] + '/py_basics')
from logger import *
from basics import *
from scipy.optimize import minimize,golden,minimize_scalar
from conf import *
import pdb

class MkvProbCalOpenPrice(object):
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

class MkvProbCalEndAve(object):
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

class MkvCalTransMat(object):
    def __init__(self,df,price,n_states):
        self.open_rtn = df[OPEN_KEY].values/price - 1.
        if n_states % 2 == 0:
            Log(LOG_FATAL) << "Num of states must be odd: {}".format(n_states)
        self.n_states = n_states

        Log(LOG_INFO) << "Trans mat prob cal created"
    def __labelMinbars(self,tid_s,tid_e,tp_return,sl_return):
        self.labels = []
        # pdb.set_trace()
        openrtn = self.open_rtn[tid_s:tid_e]
        self.drtn = (tp_return - sl_return) / self.n_states

        for rtn in openrtn:
            w = (rtn-sl_return)/self.drtn
            sid = int(np.floor(w))
            if w < 0:
                sid = self.n_states+1
            else:
                if sid > self.n_states-1:
                    sid = self.n_states
            self.labels.append(sid)

        # pdb.set_trace()
        lbs = np.array(self.labels)
        ntp = sum(lbs==self.n_states)
        nsl = sum(lbs==self.n_states+1)
        print("ntp = {}, nsl = {}".format(ntp,nsl))

    def compWinProb(self,tid_s,tid_e,tp_return,sl_return,disp=False):
        self.__labelMinbars(tid_s,tid_e,tp_return,sl_return)
        id0 = int((0 - sl_return) / self.drtn)
        freqmat  = np.zeros((self.n_states,self.n_states+2))
        for i in range(self.n_states):
            for j in range(self.n_states+2):
                freqmat[i,j] = count_subarr(self.labels,[i,j])

        print(freqmat)
        t = freqmat[:,-2]
        if sum(t) == 0:
            return 0.
        # pdb.set_trace()
        transmat,id0 = self.cleanZeroRowsCols(freqmat,id0)

        print(np.round(transmat,3),id0)
        if id0<0:
            return 0
        if transmat[id0,id0] == 1:
            return 0
        Q = transmat[:,:-2]
        I = np.identity(Q.shape[0])
        f = transmat[:,-2].reshape(-1,1)
        if sum(f) == 0: # none of states could reach tp state
            return 0.

        tmp = np.linalg.inv(I-Q)
        v = np.matmul(tmp,f)

        if disp:
            print(freqmat.astype(np.int32))
            print("prob: ",v[id][0])
        return v[id0][0]
    def cleanZeroRowsCols(self,freqmat,id0):
        to_rm = []
        for i in range(freqmat.shape[0]):
            total = sum(freqmat[i,:])
            if total == 0:
                to_rm.append(i)
            elif freqmat[i,i] == total:
                to_rm.append(i)
            else:
                pass

        fqm = freqmat.copy()
        if len(to_rm) > 0:
            fqm = np.delete(fqm,to_rm,axis=0)
            fqm = np.delete(fqm,to_rm,axis=1)
        # pdb.set_trace()
        tmp = 0
        for i in range(len(to_rm)):
            if i <= id0:
                tmp+=1
        id0-=tmp
        if id0 < 0:
            return -1,id0

        transmat = np.zeros(fqm.shape)
        for i in range(fqm.shape[0]):
            total = sum(fqm[i,:])
            transmat[i,:] = fqm[i,:] / total

        return transmat,id0

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
    if mkvconf.getProbCalType() == 0:
        mkvcal = MkvProbCalOpenPrice(df,price)
    elif mkvconf.getProbCalType() == 1:
        mkvcal = MkvCalTransMat(df,price,mkvconf.getNumStates())
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

    print("Optimized tp&sl: ", tp)

    wp = mkvcal.compWinProb(hist_start,hist_end,tp,-tp,True)

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
    # for i in range(84321-2,len(df)-1):
    tarid = 87072-2
    hist_start = tarid - hist_len
    hist_end = tarid
    price = df[OPEN_KEY][tarid]

    mkvcal = None
    if mkvconf.getProbCalType() == 0:
        mkvcal = MkvProbCalOpenPrice(df,price)
    elif mkvconf.getProbCalType() == 1:
        mkvcal = MkvCalTransMat(df,price,mkvconf.getNumStates())
    else:
        pass

    tp = mkvconf.getTPReturn()
    sl = mkvconf.getSLReturn()

    wp = mkvcal.compWinProb(hist_start,hist_end,tp,sl)

    print("tp,sl: ",tp,sl)
    print("tp prob.: ", wp)


