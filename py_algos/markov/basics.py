import pdb
from enum import IntEnum
import math
from logger import *
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import skew,kurtosis
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import statsmodels.api as sm
DATE_KEY = '<DATE>'
TIME_KEY = '<TIME>'
OPEN_KEY = '<OPEN>'
HIGH_KEY = '<HIGH>'
LOW_KEY  = '<LOW>'
CLOSE_KEY = '<CLOSE>'
TICKVOL_KEY = '<TICKVOL>'
SPREAD_KEY = '<SPREAD>'
STD_KEY  = 'STD'
MID_KEY = 'MID'
RET_KEY = "RET"
ENDDATE_KEY = "END_DATETIME"
TIMESTAMP_KEY = "TIMESTAMP"
TYNY_PROB = 0.001

class State(IntEnum):
    ORIGIN = 0,
    TP = 1,
    SL = 2,
    NONE=3

def count_subarr(arr,sub_arr):
    occ = 0
    sub_len = len(sub_arr)
    for i in range(len(arr)-len(sub_arr)):
        k=0
        for j in range(len(sub_arr)):
            if arr[i+j] == sub_arr[j]:
                k+=1
        if k==sub_len:
            occ+=1
    return occ

def golden_search_min_prob(func, args, bounds, xtol = 1e-2, maxinter = 1000):
    phi = (math.sqrt(5) - 1)*.5
    a = bounds[0]
    b = bounds[1]
    fa = func(a,*args)
    fb = func(b,*args)
    print(a,fa)
    print(b,fb)
    if fa == 0:
        return a,fa
    fk = 0
    for i in range(maxinter):
        d = phi*(b-a)
        x1 = b-d
        x2 = a+d
        f1 = func(x1,*args)
        f2 = func(x2,*args)
        print(x1,f1)
        print(x2,f2)
        fk+=2
        if f1 < f2:
            b = x2
        elif f1 > f2:
            a = x1
        elif f1 == 0 and f2 == 0:
            b = x1
        elif f1==f2:
            b = x2
        else:
            pass

        x0 = (a+b)*.5
        err = abs(a-b)/abs(x0)
        if err < xtol:
            break

    x0 = (a+b)*.5
    fmin = func(x0,*args)
    fk+=1
    Log(LOG_INFO) << "Optimization done. Function evaluations: {}".format(fk)
    return x0,fmin


class FreqCounter(object):
    def __init__(self,arr):
        self.sorted_arr = np.sort(arr)

    def __bi_sec_search(self,target,isLow):
        lw = 0
        hi = len(self.sorted_arr)-1
        if isLow and self.sorted_arr[lw] >= target:
            return lw
        if not isLow and self.sorted_arr[hi]<= target:
            return hi+1
        while hi-lw>1:
            idx = int((lw+hi)/2)
            mid = self.sorted_arr[idx]
            if mid < target:
                lw = idx
            elif mid > target:
                hi = idx
            else:
                if isLow:
                    lw = idx
                    while self.sorted_arr[lw] == target:
                        lw-=1
                    return lw+1
                else:
                    hi = idx
                    while self.sorted_arr[hi] == target:
                        hi+=1
                    return hi
        if isLow:
            return lw+1
        return hi

    def __countInRange(self,lb,ub):
        idlw = self.__bi_sec_search(lb,True)
        idhi = self.__bi_sec_search(ub,False)
        return idhi - idlw
    def compRangeProb(self,lb,ub):
        count = self.__countInRange(lb,ub)
        return count/len(self.sorted_arr)

def ns_lap_mu(x, mu, k, s):
    y = np.piecewise(x, [x<=mu], [lambda x:k*k/(1+k*k)*np.exp((x-mu)/s/k), lambda x:1-1/(1+k*k)*np.exp(-k*(x-mu)/s)])
    return y

def lap(x, mu, b):
    y = np.piecewise(x, [x<=mu], [lambda x:0.5*np.exp((x-mu)/b), lambda x:1-0.5*np.exp((-x+mu)/b)])
    return y

def lap0(x, b):
    mu = 0.
    y = np.piecewise(x, [x<=mu], [lambda x:0.5*np.exp((x-mu)/b), lambda x:1-0.5*np.exp((-x+mu)/b)])
    return y

class CDFCounter(object):
    def __init__(self,arr):
        self.sorted_arr = np.sort(arr)
        N = len(arr)
        self.x = []
        self.cdf = []
        idx = 0

        while 1:
            tar = self.sorted_arr[idx]
            self.x.append(tar)
            j = idx
            while j < N and self.sorted_arr[j]==tar:
                j+=1
            self.cdf.append(j/N)
            if j==N:
                break
            idx = j

        self.x = np.array(self.x)
        self.x = np.insert(self.x,0,-0.1)
        self.x = np.append(self.x,0.1)
        self.cdf = np.array(self.cdf)
        self.cdf = np.insert(self.cdf,0,0)
        self.cdf = np.append(self.cdf,1.)

        # fit cdf to asymmetric-laplacian distribution
        # self.fitFunc = ns_lap_mu
        # self.popt,_ = curve_fit(self.fitFunc,self.x,self.cdf)
        # self.compFitErr()
    def compFitErr(self):
        y = self.fitFunc(self.x,*self.popt)
        err = y - self.cdf
        rms = np.sqrt(np.mean(err*err))
        print("rms of cdf fit: ", rms)
        return rms
    def __compCDF(self,x):
        return self.fitFunc(x,*self.popt)
    def compCDF(self,x):
        if x >= self.sorted_arr[-1]:
            return 1.
        if x < self.sorted_arr[0]:
            return 0.
        # return np.interp(x,self.x,self.cdf)
        f = interp1d(self.x, self.cdf, kind='cubic')
        return f(x)
    def compRangeProb(self,lb,ub):
        uy = self.compCDF(ub)
        ly = self.compCDF(lb)
        return uy - ly

SKEWNESS_LIMIT=1.8
class CDFLaplace(object):
    def __init__(self,rtn0):
        rtn = np.sort(rtn0)
        sk = skew(rtn)
        l0 = len(rtn)
        sk,rtn = self.filterOutliers(rtn)
        print("Filtered elements: ", l0 - len(rtn))
        if l0 - len(rtn) > 5:
            pdb.set_trace()

        fs = lambda x: 2*(1-x**6) - sk*(x**4+1)**(3/2)
        ks = fsolve(fs,[2])
        self.kappa = ks[0]
        # print("ckeck kappa root: ", fs(self.kappa))
        if abs(fs(self.kappa)) > .1:
            pdb.set_trace()
        sd = np.std(rtn)
        # pdb.set_trace()
        self.lmb = np.sqrt((1+self.kappa**4)/self.kappa**2/sd**2)
        if abs(self.lmb) > 1e6:
            pdb.set_trace()
        self.mu = np.mean(rtn) - (1-self.kappa**2)/self.lmb/self.kappa
        print("skew = ",sk)
        print("kappa = {}, lambda = {}, mu = {}".format(self.kappa,self.lmb,self.mu))

    def filterOutliers(self,rtn):
        #pdb.set_trace()
        while 1:
            sk = skew(rtn)
            if sk < SKEWNESS_LIMIT and sk > -SKEWNESS_LIMIT:
                break
            if sk >= SKEWNESS_LIMIT:
                rtn =rtn[:-1]
            if sk <= -SKEWNESS_LIMIT:
                rtn = rtn[1:]
        return sk,rtn
    def compCDF(self,x):
        mu = self.mu
        k = self.kappa
        s = 1./self.lmb
        y = np.piecewise(x, [x <= mu], [lambda x: k * k / (1 + k * k) * np.exp((x - mu) / s / k),
                                        lambda x: 1 - 1 / (1 + k * k) * np.exp(-k * (x - mu) / s)])
        return y

    def compRangeProb(self, lb, ub):
        uy = self.compCDF(ub)
        ly = self.compCDF(lb)
        return uy - ly

class ECDFCal(object): #empirical cdf calculator
    def __init__(self,rtn0):
        self.ecdf = sm.distributions.ECDF(rtn0)
    def compCDF(self,x):
        return self.ecdf(x)
    
    def compRangeProb(self,lb,ub):
        uy = self.compCDF(ub)
        ly = self.compCDF(lb)
        return uy-ly

if __name__ == "__main__":
    import sys
    import pandas as pd
    df=pd.read_csv(sys.argv[1],sep='\t')
    p = df['<OPEN>'].values
    r = np.diff(np.log(p))
    n = 1440
    fq = FreqCounter(r[-n:])
    p = fq.compRangeProb(0,0.003)

    # arr = np.array([1,2,2,3,3,3,4,5])
    # fq = FreqCounter(arr)


