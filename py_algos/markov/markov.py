import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys, os
import copy
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
            print("ub_rtn = {}, n01 = {}, n02 = {}, tp prob. = {} ".format(tp_return, n01, n02, p01))

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
            print("ub_rtn = {}, n01 = {}, n02 = {}, tp prob. = {} ".format(tp_return, n01, n02, p01))

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
        print("effective states: ", len(self.labels) - ntp - nsl)

    def compWinProb(self,tid_s,tid_e,tp_return,sl_return,disp=False):
        self.__labelMinbars(tid_s,tid_e,tp_return,sl_return)
        id0 = int((0 - sl_return) / self.drtn)
        freqmat  = np.zeros((self.n_states,self.n_states+2))
        for i in range(self.n_states):
            for j in range(self.n_states+2):
                freqmat[i,j] = count_subarr(self.labels,[i,j])
            # if freqmat[i,i-1] == 0:
            #     freqmat[i,i-1] = 1
            # if freqmat[i,i+1] == 0:
            #     freqmat[i,i+1] = 1

        print(freqmat)
        t = freqmat[:,-2]
        if sum(t) == 0:
            return 0.
        # pdb.set_trace()
        transmat,id0 = self.cleanZeroRowsCols(freqmat,id0)

        print(np.round(transmat,3),id0)

        Q = transmat[:,:-2]
        I = np.identity(Q.shape[0])
        f = transmat[:,-2].reshape(-1,1)
        if sum(f) == 0: # none of states could reach tp state
            return 0.

        if np.linalg.det(I-Q) == 0:
            pdb.set_trace()

        tmp = np.linalg.inv(I-Q)
        v = np.matmul(tmp,f)

        if disp:
            print(freqmat.astype(np.int32))
            print("prob: ",v[id][0])
        return v[id0][0]
    def cleanZeroRowsCols(self,freqmat,id0):
        # to_rm = []
        # for i in range(freqmat.shape[0]):
        #     total = sum(freqmat[i,:])
            # if total == 0:
            #     to_rm.append(i)
            # elif freqmat[i,i] == total:
            #     to_rm.append(i)
            # else:
            #     pass

        # fqm = freqmat.copy()
        # if len(to_rm) > 0:
        #     fqm = np.delete(fqm,to_rm,axis=0)
        #     fqm = np.delete(fqm,to_rm,axis=1)
        # # pdb.set_trace()
        # tmp = 0
        # for i in range(len(to_rm)):
        #     if i <= id0:
        #         tmp+=1
        # id0-=tmp
        # if id0 < 0:
        #     return -1,id0
        fqm = freqmat.copy()
        transmat = np.zeros(fqm.shape)
        for i in range(fqm.shape[0]):
            total = sum(fqm[i,:])
            if total == 0:
                transmat[i,i-1] = .5
                transmat[i,i+1] = .5
            else:
                transmat[i,:] = fqm[i,:] / total
                if transmat[i,i+1] == 0:
                    transmat[i,i+1] = TYNY_PROB
                    transmat[i,i] -= TYNY_PROB
                if transmat[i,i-1] == 0:
                    transmat[i,i-1] = TYNY_PROB
                    transmat[i,i] -= TYNY_PROB

        return transmat,id0

class MkvCalEqnSol(object):
    def __init__(self,df, npts):
        self.n_partitions = npts
        self.df = df
        self.rtn=None

    def getRtn(self):
        return self.rtn
    
    def compTransMat_noabsorb(self,rtns,ub_rtn,lb_rtn):#no absorb states, but cannot go higher or lower
        probcal = ECDFCal(rtns)
        n_states = self.n_partitions+2
        d = (ub_rtn-lb_rtn)/self.n_partitions
        idxDiff2Prob = {}
        for i in range(-n_states-1,n_states+2):
            p = probcal.compRangeProb(i*d-d/2,i*d+d/2)
            idxDiff2Prob[i] = p
        pm = np.zeros((n_states,n_states))
        
        for i in range(n_states):
            for j in range(n_states):      
                if j==0:
                    dr = (j-i)*d+d*.5
                    pm[i,j] = probcal.compRangeProb(-.5, dr)
                elif j==n_states-1:
                    dr = (j-i)*d - d*.5
                    pm[i,j] = probcal.compRangeProb(dr, .5)
                else:
                    pm[i,j] = idxDiff2Prob[j-i]
        
        # truncate at end states
        s = np.sum(pm[0,:])
        pm[0,0] += (1-s)
        s = np.sum(pm[-1,:])
        pm[-1,-1] += (1-s)
        
        return pm
    def compLimitDist(self,rtns,ub_rtn,lb_rtn): #stationary distribution of states
        pm = self.compTransMat_noabsorb(rtns, ub_rtn, lb_rtn)
        npts = self.n_partitions+2
        I = np.identity(npts)
        ONE = np.ones((npts,npts))
        one = np.ones((1,npts))
        tmp = I - pm + ONE
        
        tmp = np.linalg.inv(tmp)
        
        dist = np.matmul(one,tmp)
        return dist
    
    def compLimitRtn(self,tid_s,tid_e,ub_rtn,lb_rtn): # expected return per cdf 
        pc = self.df[OPEN_KEY][tid_s:tid_e+1]
        rtns = np.diff(np.log(pc))
        dist = self.compLimitDist(rtns, ub_rtn, lb_rtn)
        d = (ub_rtn-lb_rtn)/self.n_partitions
        s = 0.
        for i in range(self.n_partitions+2):
            s += dist[0,i]*(lb_rtn-d/2 + i*d)
        return s
    
    def auxExpect1HitStep(self,pm,tar_state,lookfwd):
        pf = copy.deepcopy(pm)
        f = copy.deepcopy(pf[:,tar_state])
        pf[:,tar_state] = 0
        pb = f[int(len(f)/2)]
        
        expstep=pb
        for i in range(lookfwd):
            f = np.matmul(pf,f)
            pb = f[int(len(f)/2)]
            expstep += (i+2)*pb
            
        return expstep
            
    def compExpectHitSteps(self,tid_s,tid_e,ub_rtn,lb_rtn,lookfwd):
        pc = self.df[OPEN_KEY][tid_s:tid_e+1]
        rtns = np.diff(np.log(pc))
        pm = self.compTransMat_absorb(rtns, ub_rtn, lb_rtn)
        
        sp_up = self.auxExpect1HitStep(pm,-1,lookfwd)
        sp_dn = self.auxExpect1HitStep(pm, 0,lookfwd)
        
        return sp_up,sp_dn,sp_up/sp_dn
        
        
    def compTransMat_absorb(self,rtns,ub_rtn,lb_rtn):
        probcal = ECDFCal(rtns)
        npts = self.n_partitions
        d = (ub_rtn-lb_rtn)/npts
        idxDiff2Prob = {}
        for i in range(-npts-1,npts+2):
            p = probcal.compRangeProb(i*d-d/2,i*d+d/2)
            idxDiff2Prob[i] = p
        
        pm = np.zeros((npts+2,npts+2))
        for i in range(npts+2):
            for j in range(npts+2):
                if j==0:
                    dr = (j-i)*d+d*.5
                    pm[i,j] = probcal.compRangeProb(-.5, dr)
                elif j==npts+1:
                    dr = (j-i)*d - d*.5
                    pm[i,j] = probcal.compRangeProb(dr, .5)
                else:
                    pm[i,j] = idxDiff2Prob[j-i]
        pm[0,:] = 0
        pm[0,0] = 1.
        pm[-1,:] = 0
        pm[-1,-1] = 1.
        return pm
            
    def compWinProb(self,tid_s,tid_e,ub_rtn,lb_rtn):
        pc = self.df[OPEN_KEY][tid_s:tid_e+1]
        self.rtn = np.diff(np.log(pc))
        
        rtn = copy.deepcopy(self.rtn)
        # print("Ave rtn: ",np.mean(rtn))
        # self.transProbCal = CDFCounter(rtn)
        # self.transProbCal = CDFLaplace(rtn)
        # pdb.set_trace()
        self.transProbCal = ECDFCal(rtn)
        # print("ECDFCal is used")
        npts = self.n_partitions
        d = (ub_rtn-lb_rtn)/npts
        idxDiff2Prob = {}
        for i in range(-npts+1,npts):
            p = self.transProbCal.compRangeProb(i*d-d/2,i*d+d/2)
            idxDiff2Prob[i] = p

        C = np.zeros((npts,npts))
        I = np.identity(npts)
        Q = np.zeros((npts,1))
        one = np.ones((npts,1))
        # Qsell = np.zeros((npts,1))

        for i in range(npts):
            for j in range(npts):
                C[i,j] = idxDiff2Prob[j-i]
            Q[i] = self.transProbCal.compRangeProb((npts-i)*d-d/2,.5)
            # Qsell[i] = self.transProbCal.compRangeProb(-1,(-1-i)*d+d/2)

        # pdb.set_trace()
        # print(C)
        tmp = I-C
        # det = np.linalg.det(tmp)
        # if np.linalg.det(tmp) == 0:
        #     pdb.set_trace()
        try:
            tmp = np.linalg.inv(tmp)
        except:
            print("inversion fails")
            return 0.5

        pr = np.matmul(tmp,Q)
        # ps = np.matmul(tmp,Qsell)
        steps = np.matmul(tmp,one)

        idx = int((0-lb_rtn)/d)
        # print("Expected buy tp steps",steps[idx][0])
        # print("pr = {}, det = {}".format(pr[idx][0],det))
        # return pr[idx][0],ps[idx][0],steps[idx][0]
        p = pr[idx][0]
        sp = steps[idx][0]
        
        
        if p < 0:
            p = 0.
        return p,sp

class FirstHitProbCal(object):
    def __init__(self,df,npts):
        self.n_partitions = npts
        self.df = df
    def comp1stHitProb(self,tid_s, tid_e, ub_rtn,lb_rtn, steps):
        pc = self.df[OPEN_KEY][tid_s:tid_e + 1]
        rtn = np.diff(np.log(pc))
        print("Ave rtn: ", np.mean(rtn))
        # self.transProbCal = CDFCounter(rtn)
        self.transProbCal = CDFLaplace(rtn)
        npts = self.n_partitions
        d = (ub_rtn - lb_rtn) / npts
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p

        PT = np.zeros((npts+2,npts+2))
        for i in range(npts):
            for j in range(npts):
                PT[i, j] = idxDiff2Prob[j - i]
            PT[i,npts] = self.transProbCal.compRangeProb((npts - i) * d - d / 2, .5)
            PT[i,npts+1] = self.transProbCal.compRangeProb(-0.5,(-1-i)*d+d/2)

        PT[npts,npts] = 1.
        PT[npts+1,npts+1] = 1.
        # pdb.set_trace()
        idc = int((0-lb_rtn)/d)
        pup = self.firstHitProb(PT,steps,idc,npts)
        pdw = self.firstHitProb(PT,steps,idc, npts+1)

        return pup,pdw
    def firstHitProb(self,PT,steps, s_id, tar_id):
        f = PT[:,tar_id].copy()
        P = PT.copy()
        pbs = []
        pbs.append(f[s_id])
        P[:,tar_id]=0
        for i in range(steps):
            f = np.matmul(P,f)
            pbs.append(f[s_id])
        return sum(pbs)

class RSICal(object):
    def __init__(self,df):
        self.df = df
    def compWinProb(self,hist_start,hist_end):
        p = self.df['<OPEN>'].values[hist_start:hist_end+1]
        r = np.diff(np.log(p))
        rp = r[r>=0]
        rn = r[r<0]
        mrp = np.mean(rp)
        mrn = abs(np.mean(rn))
        return (mrp/(mrp+mrn))

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

    tp = mkvconf.getUBReturn()
    sl = mkvconf.getLBReturn()

    wp = mkvcal.compWinProb(hist_start,hist_end,tp,sl)

    print("tp,sl: ",tp,sl)
    print("tp prob.: ", wp)


