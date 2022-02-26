from enum import IntEnum
import math
from logger import *
import numpy as np
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


