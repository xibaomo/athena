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


class MkvAbsorbCal(object):
    def __init__(self,nstates,cdf='emp'):
        self.n_states = nstates
        self.cdfType = cdf
        self.transProbCal = None
    def createCDFCal(self,rtns):
        if self.cdfType == 'emp':
            self.transProbCal = ECDFCal(rtns)
        elif self.cdfType == 'gauss':
            self.transProbCal = GaussCDFCal(rtns)
        elif self.cdfType == "laplace":
            self.transProbCal = LaplaceCDFCal(rtns)
        else:
            print("Wrong cdf type: ", self.cdfType)
            sys.exit(1)
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

