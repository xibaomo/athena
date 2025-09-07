import time

import statsmodels.api as sm
import numpy as np
from scipy.stats import norm
import math
import sys,os
import cupy as cp
import pdb

class ECDFCal(object):  # empirical cdf calculator
    def __init__(self, rtn0):
        self.ecdf = sm.distributions.ECDF(rtn0)

    def compCDF(self, x):
        return self.ecdf(x)

    def compRangeProb(self, lb, ub):
        uy = self.compCDF(ub)
        ly = self.compCDF(lb)
        return uy - ly
class WeightedCDFCal(object): # cdf calculator: weighted sum of recent days
    def __init__(self,rtn0,wts,spacing):
        self.wts = wts
        self.ecdfs = []
        # breakpoint()
        for i in range(len(wts)):
            eid = -spacing*(len(wts)-i-1)
            if eid == 0:
                eid = -1
            cdf = sm.distributions.ECDF(rtn0[-spacing*(len(wts)-i):eid])
            self.ecdfs.append(cdf)
    def compCDF(self,x):
        weighted_cdf = [c(x)*w for c,w in zip(self.ecdfs,self.wts)]
        return np.sum(weighted_cdf)
    def compRangeProb(self, lb, ub):
        uy = self.compCDF(ub)
        ly = self.compCDF(lb)
        return uy - ly
class GaussCDFCal(object):
    def __init__(self,rtn0):
        self.mean = np.mean(rtn0)
        self.std  = np.std(rtn0)
    def compCDF(self,x):
        return norm.cdf(x,loc=self.mean,scale=self.std)
    def compRangeProb(self,lb,ub):
        return self.compCDF(ub) - self.compCDF(lb)

class LaplaceCDFCal(object):
    def __init__(self,rtn0):
        self.mean = np.mean(rtn0)
        self.scale = np.std(rtn0)/math.sqrt(2.)
    def compCDF(self,x):
        if x <= self.mean:
            return .5*np.exp((x-self.mean)/self.scale)
        return 1-.5*np.exp((self.mean-x)/self.scale)
    def compRangeProb(self,lb,ub):
        return self.compCDF(ub) - self.compCDF(lb)

class MkvRegularCal(object):
    def __init__(self,nstates,cdf_cal):
        self.n_states = nstates
        self.transProbCal = cdf_cal
    def buildTransMat(self,rtns,lb_rtn,ub_rtn):
        rtns = rtns[~np.isnan(rtns)]
        # self.transProbCal = ECDFCal(rtns)
        npts = self.n_states
        d = (ub_rtn - lb_rtn) / (npts-1)
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p
        P = np.zeros((npts, npts))

        for i in range(npts):
            for j in range(npts):
                P[i, j] = idxDiff2Prob[j - i]

        # normalization
        for i in range(npts):
            s = 0.
            for j in range(npts):
                s += P[i, j]
            if s == 0:
                pdb.set_trace()
            P[i, :] = P[i, :] / s
        self.transMat = P.copy()
        return P
    def compMultiStepProb(self,steps, rtns,lb_rtn,ub_rtn):
        P = self.buildTransMat(rtns,lb_rtn,ub_rtn)
        drtn = (ub_rtn-lb_rtn)/(self.n_states-1)
        PWP = np.linalg.matrix_power(P,steps)
        idx = int((0-lb_rtn)/drtn)
        if idx < 0:
            pdb.set_trace()
        return PWP[idx,:]

class MkvAbsorbCal(object):
    def __init__(self,nstates,cdf_cal):
        self.n_states = nstates
        # self.cdfType = cdf
        self.transProbCal = cdf_cal
        # self.cdf_wts=cdf_wts
    # def createCDFCal(self,rtns):
    #     if self.cdfType == 'emp':
    #         self.transProbCal = ECDFCal(rtns)
    #     elif self.cdfType == 'gauss':
    #         self.transProbCal = GaussCDFCal(rtns)
    #     elif self.cdfType == "laplace":
    #         self.transProbCal = LaplaceCDFCal(rtns)
    #     elif self.cdfType == "wts":
    #         self.transProbCal = WeightedCDFCal(rtns,self.cdf_wts)
    #     else:
    #         print("Wrong cdf type: ", self.cdfType)
    #         sys.exit(1)
    def buildTransMat(self,lb_rtn,ub_rtn):
        # rtns = rtns[~np.isnan(rtns)]
        # self.createCDFCal(rtns)
        npts = self.n_states
        d = (ub_rtn - lb_rtn) / npts
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p
        P = np.zeros((npts+2,npts+2))

        for i in range(npts):
            for j in range(npts):
                P[i, j] = idxDiff2Prob[j - i]
        for i in range(npts):
            P[i,npts] = self.transProbCal.compRangeProb((npts - i) * d - d / 2, 1.)
            P[i,npts+1] = self.transProbCal.compRangeProb(-1., (-1-i)*d + d/2)
        P[npts,npts] = 1.
        P[npts+1,npts+1] = 1.

        # normalization
        for i in range(npts+2):
            s = 0.
            for j in range(npts+2):
                s += P[i,j]
            if s == 0:
                pdb.set_trace()
            P[i,:] = P[i,:]/s
        self.transMat = P.copy()
        return P
    def comp1stHitProb(self,steps,start_id, tar_id1,tar_id2):
        P1 = self.transMat.copy()
        P2 = self.transMat.copy()
        f1 = P1[:,tar_id1].copy()
        f2 = P2[:,tar_id2].copy()
        P1[:,tar_id1] = 0
        P2[:,tar_id2] = 0

        mid = start_id
        pb1 = f1[mid]
        pb2 = f2[mid]
        # pdb.set_trace()
        for i in range(steps-1):
            f1 = P1 @ f1
            f2 = P2 @ f2
            pb1 += f1[mid]
            pb2 += f2[mid]
        return pb1,pb2
    def compWinProb(self, rtns, lb_rtn, ub_rtn):
        rtns = rtns[~np.isnan(rtns)]
        self.createCDFCal(rtns)
        npts = self.n_states
        d = (ub_rtn - lb_rtn) / npts
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p

        C = np.zeros((npts, npts))
        I = np.identity(npts)
        Q = np.zeros((npts, 1))
        one = np.ones((npts, 1))
        # Qsell = np.zeros((npts,1))

        for i in range(npts):
            for j in range(npts):
                C[i, j] = idxDiff2Prob[j - i]
            Q[i] = self.transProbCal.compRangeProb((npts - i) * d - d / 2, .5)

        tmp = I - C
        tic = time.time()
        # pdb.set_trace()
        try:
            tmp = cp.array(tmp)
            tmp = cp.linalg.inv(tmp)
            tmp = cp.asnumpy(tmp)
        except:
            pdb.set_trace()
            print("inversion fails")
            return 0.5,1e10

        pr = np.matmul(tmp, Q)
        # ps = np.matmul(tmp,Qsell)
        steps = np.matmul(tmp, one)

        # print("matrix ops take: ", time.time()-tic)
        idx = int((0 - lb_rtn) / d)
        p = pr[idx][0]
        sp = steps[idx][0]

        if p < 0:
            p = 0.
        return p, sp

    def compUpAveSteps(self,rtns,lb,ub):
        rtns = rtns[~np.isnan(rtns)]
        self.createCDFCal(rtns)
        npts = self.n_states
        d = (ub - lb) / npts
        idxDiff2Prob = {}
        for i in range(-npts + 1, npts):
            p = self.transProbCal.compRangeProb(i * d - d / 2, i * d + d / 2)
            idxDiff2Prob[i] = p

        C = np.zeros((npts, npts))
        I = np.identity(npts)
        Q = np.zeros((npts, 1))
        one = np.ones((npts, 1))
        # Qsell = np.zeros((npts,1))

        for i in range(npts):
            for j in range(npts):
                C[i, j] = idxDiff2Prob[j - i]
            Q[i] = self.transProbCal.compRangeProb((npts - i) * d - d / 2, .5)
        # normalize C and Q
        for i in range(npts):
            s = np.sum(C[i,:]) + Q[i]
            if s == 0:
                pdb.set_trace()
            C[i,:] = C[i,:]/s
            Q[i] = Q[i]/s

        tmp = I - C
        try:
            tmp = np.linalg.inv(tmp)
        except:
            print("inversion fails")
            return  1e10

        steps = np.matmul(tmp, one)

        idx = int((0 - lb) / d)
        sp = steps[idx][0]

        return sp


def dp_minimize(candidates,cal_f, max_n_choose=-1, min_n_choose=10, result_rank=0):
    '''
    candidates: [(sym1,params1),(sym2,params2),...]
    n_choose: number of selections
    cal_f: function to compute cost
    '''

    n = len(candidates)
    k = max_n_choose
    if k < 0:
        k = n
    dp = [[ 1e20 for _ in range(k+1)] for _ in range(n+1)]
    chosen = [[[] for _ in range(k+1)] for _ in range(n+1)]

    # build the table bottom-up
    for i in range(1,n+1):
        # print("checking {}th sym out of {}".format(i, n))
        for j in range(1,k+1):
            if j<=i:
                args = chosen[i-1][j-1] + [candidates[i-1]]
                cost,_ = cal_f(args)
                if cost < dp[i-1][j]:
                    dp[i][j] = cost
                    chosen[i][j] = args
                else:
                    dp[i][j] = dp[i-1][j]
                    chosen[i][j] = chosen[i-1][j]

    # min_idx = dp[n].index(min(dp[n]))
    # return dp[n][min_idx],chosen[n][min_idx]
    arr = np.array(dp[n][min_n_choose:])
    sorted_id = np.argsort(arr)
    idx = sorted_id[result_rank]
    minval = arr[idx]
    choice = chosen[n][idx+min_n_choose]
    return minval,choice

def __compProb1stHidBounds(rtns, steps,ub_rtn=0.05,lb_rtn=-0.05):
    d=0.001/8
    ns = int((ub_rtn-lb_rtn)/d)
    mkvcal = MkvAbsorbCal(ns)
    # rtns = df['Close'].pct_change().values[-lookback:]
    # pdb.set_trace()
    # print(f"{len(rtns)}-tradehour volatility: {np.std(rtns):.5f}")
    mkvcal.buildTransMat(rtns,lb_rtn,ub_rtn)
    # d = (ub_rtn-lb_rtn)/ns
    mid = int((0-lb_rtn)/d)
    pbu,pbd = mkvcal.comp1stHitProb(steps,mid,-2,-1)
    return pbu,pbd

def compProb1stHitBounds(steps, cdf_cal, ub_rtn=.5,lb_rtn=-.5):
    d = 0.001/4
    ns = int((ub_rtn-lb_rtn)/d)
    # breakpoint()
    mkvcal = MkvAbsorbCal(ns,cdf_cal)
    mkvcal.buildTransMat(lb_rtn,ub_rtn)
    mid = int((0-lb_rtn)/d)
    pbu, pbd = mkvcal.comp1stHitProb(steps, mid, -2, -1)
    return pbu, pbd

def compMultiStepProb(rtns,steps,lb_rtn,ub_rtn, cdf_cal,drtn=0.001/4):

    ns = int((ub_rtn-lb_rtn)/drtn)
    mkvcal = MkvRegularCal(ns,cdf_cal)
    P = mkvcal.buildTransMat(rtns, lb_rtn, ub_rtn)

    # pdb.set_trace()
    PWP = np.linalg.matrix_power(P, steps)
    idx = int((0 - lb_rtn) / drtn)
    if idx < 0:
        pdb.set_trace()
    return PWP[idx, :]

def compute_steady_dist(rtns,lb_rtn,ub_rtn):
    drtn = 0.001 / 4
    ns = int((ub_rtn - lb_rtn) / drtn)
    mkvcal = MkvRegularCal(ns)
    P = mkvcal.buildTransMat(rtns, lb_rtn, ub_rtn)

    I = np.eye(ns)
    ONE = np.ones((ns,ns))
    pi = np.ones((1,ns))
    tmp = np.eye(ns) - P + ONE
    tmp = np.linalg.inv(tmp)
    pi = pi@tmp
    return pi.ravel()
