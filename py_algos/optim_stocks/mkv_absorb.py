import statsmodels.api as sm
import numpy as np
from scipy.stats import norm
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

class MkvAbsorbCal(object):
    def __init__(self,nstates,cdf='emp'):
        self.n_states = nstates
        self.cdfType = cdf
        self.transProbCal = None

    def compWinProb(self, rtns, lb_rtn, ub_rtn):
        rtns = rtns[~np.isnan(rtns)]
        if self.cdfType == 'emp':
            self.transProbCal = ECDFCal(rtns)
        elif self.cdfType == 'gauss':
            self.transProbCal = GaussCDFCal(rtns)
        else:
            print("Wrong cdf type: ", self.cdfType)
        # print("ECDFCal is used")
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

        # pdb.set_trace()
        # print(C)
        tmp = I - C

        try:
            tmp = np.linalg.inv(tmp)
        except:
            print("inversion fails")
            return 0.5,1e10

        pr = np.matmul(tmp, Q)
        # ps = np.matmul(tmp,Qsell)
        steps = np.matmul(tmp, one)

        idx = int((0 - lb_rtn) / d)
        p = pr[idx][0]
        sp = steps[idx][0]

        if p < 0:
            p = 0.
        return p, sp