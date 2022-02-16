from enum import IntEnum
import math
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

def golden_search_min(func, args, bounds, xtol = 1e-3, maxinter = 1000):
    phi = (math.sqrt(5) - 1)*.5
    a = bounds[0]
    b = bounds[1]
    for i in range(maxinter):
        d = phi*(b-a)
        x1 = b-d
        x2 = a+d
        f1 = func(x1,*args)
        f2 = func(x2,*args)
        if f1 < f2:
            b = x2
        else:
            a = x1
        if abs(a-b) < xtol:
            break

    x0 = (a+b)*.5
    fmin = func(x0,*args)
    return x0,fmin