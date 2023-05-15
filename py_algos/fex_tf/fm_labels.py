#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:40:32 2022

@author: naopc
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.stats import anderson_ksamp
from scipy.stats import ks_2samp
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import entropy
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

mkv_path = os.environ['ATHENA_HOME']+'/py_algos/markov'
sys.path.append(mkv_path)
mkv_path = os.environ['ATHENA_HOME']+'/py_algos/fex_tf'
sys.path.append(mkv_path)
mkv_path = os.environ['ATHENA_HOME'] + '/api/api_athena/pyapi'
sys.path.append(mkv_path)
from fexconf import *
from markov import *
from feature import *
from athena import *

TM_KEY = 'DATETIME'
def compare_dist1(a1, a2):
    test_statistic, pv = ks_2samp(a1, a2)
    if pv > 0.05:
        return 1
    return 0

def compare_dist2(a1, a2):
    result = anderson_ksamp([a1, a2])
    if  result.significance_level > 0.05:
        return 1
    return 0

def isSharpHour(tm):
    if tm.minute == 0 and tm.second == 0:
        return True

    return False

def find_range(df, stm, ndays):
    tms = df['<DATE>'] + " " + df['<TIME>']
    tms = pd.to_datetime(tms)
    df[TM_KEY] = tms

    sid = bisec_time_point(df, stm)
    stm = df[TM_KEY][sid]
    print("Starting time set to ",stm)

    etm = stm + pd.Timedelta(ndays + ' days')
    eid = bisec_time_point(df, etm)
    etm = df[TM_KEY][eid]
    print("Ending time set to ",etm)

    return sid, eid

def bisec_time_point(df, tm):
    # pdb.set_trace()
    lb = 0
    ub = len(df)-1
    fl = df[TM_KEY][lb] - tm
    fr = df[TM_KEY][ub] - tm
    if fl.total_seconds() > 0:
        print("time {} is earlier than the 1st time point, return the 1st".format(df[TM_KEY][lb]))
        return 0
    if fr.total_seconds() <=0:
        print("time {} is newer than the latest history, return the last time".format(df[TM_KEY][-1]))
        return ub
    while ub-lb > 1:
        mid = int((lb+ub)/2)
        fm = df[TM_KEY][mid] - tm
        if fm.total_seconds() == 0:
            return mid
        if fm.total_seconds() < 0:
            lb = mid
        else:
            ub = mid

    return ub

def __find_range(df, stm, ndays):
    tms = df['<DATE>'] + " " + df['<TIME>']
    tms = pd.to_datetime(tms)
    df[TM_KEY] = tms
    etm = stm + pd.Timedelta(ndays + ' days')

    sid = -1
    for i in range(len(tms)):
        dt = tms[i] - stm
        if dt.total_seconds() == 0:
            sid = i
            break
    if sid < 0:
        return -1, -1

    eid = sid+1

    while eid < len(df):
        dt = etm - df[TM_KEY][eid]
        if dt.total_seconds() <0:
            break
        eid+=1

    return sid, eid

def __labelHours(df, sid, eid, rtn, lifetime_days):
    print('labeling range: {} minutes ... '.format(eid-sid))
    labels = []
    tid= []
    for i in range(sid, eid):
        tm = df[TM_KEY][i]
        if not isSharpHour(tm):
            continue
        idx = i+1
        p0 = df['<OPEN>'][i]
        lb = 0
        tid.append(i)
        while idx < len(df):
            rh = df['<HIGH>'][idx]/p0 - 1
            rl = df['<LOW>'][idx] /p0 - 1

            if rh > rtn and rl < -rtn:
                lb = 3
                break
            elif rh > rtn:
                lb = 1
                break
            elif rl < -rtn:
                lb = 2
                break
            else:
                dt = df[TM_KEY][idx] - df[TM_KEY][i]
                if dt.total_seconds() >= 3600*24*lifetime_days:
                    if rl > 0:
                        lb = 1
                    if rh < 0:
                        lb = 2

                    break
                pass
            idx+=1
        labels.append(lb)

    labels = np.array(labels)
    tid = np.array(tid)
    n0 = np.sum(labels==0)
    n1 = np.sum(labels==1)
    n2 = np.sum(labels==2)
    n3 = np.sum(labels==3)
    print("Labels dist: 0 - {}, 1 - {}, 2 - {}, 3 - {}".format(n0, n1, n2, n3))
    idx = labels > 0
    labels = labels[idx]
    tid = tid[idx]

    return tid, labels

def ___labelHours(df, sid, eid, rtn, lifetime_days):
    hourid, labels = athn_label_hours(df, sid, eid, rtn, lifetime_days)
    return hourid, labels

def labelHours(df, thd, lookfwd):
    ismark = df['<TIME>'].isin(['04:00:00'])
    ismark = df['<TIME>'].isin(['04:00:00','08:00:00','12:00:00','16:00:00','20:00:00'])
    hourids = [i for i, d in enumerate(ismark.values) if d]
    
    print("marked hours: ",len(hourids))
    
    hourids = np.array(hourids)
    op = df['<OPEN>'].values
    hi = df['<HIGH>'].values
    lw = df['<LOW>'].values

    #pdb.set_trace()
    labels = np.zeros(len(hourids))

    for t in range(len(hourids)):
        tid = hourids[t]
        p0 = op[tid]
        for i in range(lookfwd):
            idx = tid + i
            if idx >= len(df):
                break
            rh = hi[idx]/p0 - 1
            rl = lw[idx]/p0 - 1
            if rh >= thd:
                labels[t] = 1
                #pdb.set_trace()
                break
            if rl <= -thd:
                labels[t] = 2
                #pdb.set_trace()
                break
        if labels[t] == 4:
            idx = tid+1440-1
            idx = idx if idx < len(df) else len(df)-1
            labels[t] = 1 if op[idx]>p0 else 2

    return hourids[10:], labels[10:]
def comp_clt(df, tid, lookback, lookfwd, thd):
    ps = df['<OPEN>'].values[tid-lookback:tid+1]
    rtn = np.diff(np.log(ps))
    mu = np.mean(rtn)
    s = np.std(rtn)
    zsp = (thd - lookfwd*mu)/np.sqrt(lookfwd)/s
    zsn = (-thd - lookfwd*mu)/np.sqrt(lookfwd)/s
    prob_up = 1-norm.cdf(zsp)
    prob_dw = norm.cdf(zsn)

    return prob_up, prob_dw

def plot_labels(ffm, flbs):
    for i in range(len(flbs)):
        #spd = abs(ffm[i, 1])
        spd =(ffm[i, 1])

        if flbs[i] == 0:
            plt.plot(ffm[i, 0], spd, 'gs')
        if flbs[i] == 1:
            plt.plot(ffm[i, 0], spd, 'go',fillstyle='none')
        if flbs[i] == 2:
            plt.plot(ffm[i, 0], spd, 'rx')
        if flbs[i] == 3:
            plt.plot(ffm[i, 0], spd, 'd')
def compRSI(op, tid, lookback):
    a = np.diff(op[tid-lookback:tid])
    up = a[a>0].sum()
    dw = -a[a<=0].sum()
    return up/(up+dw)
def compRiseRatio(op, tid, lookback):
    a = np.diff(op[tid-lookback:tid])
    idx = a > 0
    return sum(idx)/len(a)

def find_turning_points(data, threshold):
    """
    Find the turning points in a time series.

    Parameters:
    ----------
    data : ndarray
        The input time series.
    max_rel_change : float
        If specified, merge adjacent turning points whose absolute relative change is below this threshold.
    min_distance : int
        If specified, merge adjacent turning points that are closer than this distance.

    Returns:
    -------
    turning_points : ndarray
        The indices of the turning points in the input data.
    """

    data = savgol_filter(data, 60*6, 2)

    # Find local minima and maxima
    local_minima = argrelextrema(data, np.less)[0]
    local_maxima = argrelextrema(data, np.greater)[0]
    turning_points = np.sort(np.concatenate([local_minima, local_maxima]))

    # drop points of small change
    ftp = [0]
    for i in range(0, len(turning_points)):
        if abs((data[ftp[-1]] - data[turning_points[i]]) / data[ftp[-1]]) > threshold:
            ftp.append(turning_points[i])
    if abs((data[ftp[-1]] - data[turning_points[-1]]) / data[ftp[-1]]) > threshold:
        ftp.append(len(data)-1)

    # only keep turning points of turning ponts
    ftp1 = [ftp[0]]
    for i in range(1, len(ftp)-1):
        if ( data[ftp[i]]-data[ftp[i-1]]) * (data[ftp[i+1]]-data[ftp[i]]) < 0:
            ftp1.append(ftp[i])
    ftp1.append(ftp[-1])
    return data, ftp1
def compRiseRatio(data): # percentage of rising time
    _, tps = find_turning_points(data, 0.0015)
    up = 0
    down = 0
    for i in range(1, len(tps)):
        if data[tps[i]] > data[tps[i-1]]:
            up+= (tps[i]-tps[i-1])
        else:
            down+= (tps[i]-tps[i-1])
    return up/(up+down)

def compSpeedRatio(data):
    smt, tps = find_turning_points(data, 0.0015)
    upspd=[]
    dwspd=[]
    for i in range(1, len(tps)):
        d = smt[tps[i]] - smt[tps[i-1]]
        t = tps[i] - tps[i-1]
        spd = d/t
        if spd > 0:
            upspd.append(spd)
        else:
            dwspd.append(-spd)

    return np.mean(upspd)/np.mean(dwspd)

# Define a function to calculate the ATR
def calc_atr(df, tid, lookback):
    high_low = df['<HIGH>'] - df['<LOW>']
    high_close_prev = abs(df['<HIGH>'] - df['<CLOSE>'].shift())
    low_close_prev = abs(df['<LOW>'] - df['<CLOSE>'].shift())
    ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis = 1)
    #pdb.set_trace()
    true_range = np.max(ranges, axis = 1)
    atr = true_range.rolling(lookback).mean()
    return atr[tid]

import statsmodels.api as sm

def compare_cdf(*datas):
    # Calculate the empirical CDF for each data set
    plt.figure()
    k = 1
    for data in datas:
        ecdf1 = sm.distributions.ECDF(data)
        x1 = np.linspace(min(data), max(data))
        y1 = ecdf1(x1)
        #plt.step(x1, y1, label='data '+str(k))
        plt.plot(ecdf1.x, ecdf1.y, '.-')
        k+=1

    plt.xlabel('Data')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    plt.legend()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} <csv> <mkv.yaml> ".format(sys.argv[0]))
        sys.exit(0)

    csvfile = sys.argv[1]
    ymlfile = sys.argv[2]

    df = pd.read_csv(csvfile, sep='\t')
    fexconf =FexConfig(ymlfile)
    op = df['<OPEN>'].values

    rtn = fexconf.getUBReturn()

    st = time.perf_counter()
    thd = 0.01
    lookfwd = 1440*5
    hourids, labels = labelHours(df, thd, lookfwd)
    
    lookback =fexconf.getLookback()
    lookfwd = fexconf.getLookforward()

    print("labeling time (s) ",time.perf_counter()-st)
    print("no lables: ",sum(labels==0))
    
    idx = hourids >= lookback
    hourids = hourids[idx]
    labels  = labels[idx]
    

    idx = labels!=0
    hourids = hourids[idx]
    labels  = labels[idx]
    #pdb.set_trace()
    hourids = hourids.astype(np.int32)  

    ##### markov
    mkvcal = MkvCalEqnSol(df, fexconf.getNumPartitions())

    
    fm = np.zeros((len(labels), 7))
    print("computing features...")
    for i in range(len(hourids)):
        st = time.perf_counter()
        tid = hourids[i]
        # hist_end = hourids[i] + 1
        # hist_start = hist_end - lookback
        # prop, sp = mkvcal.compWinProb(hist_start, hist_end, rtn, -rtn)
        # fm[i, 0] = prop
        # fm[i, 1] = rtn/sp

        #ops = df['<OPEN>'].values[tid-lookback:tid+1]
        #rts = np.diff(np.log(ops))
        #fm[i, 0] = sum(rts)
        #fm[i, 1] = np.std(rts)
        ft = compMkvFeatures(df, hourids[i], mkvcal, fexconf)
        fm[i, 0] = ft[0][0]
        fm[i, 1] = ft[0][1]
        #pu, pd = comp_clt(df, tid, lookback, lookfwd, thd)
        #pu, pd = comp_clt(df, tid, lookback, 1440, thd/2)
        arr = op[tid-lookback:tid+1]
        darr = np.diff(np.log(arr))
        fm[i, 2] = np.sum(darr)
        fm[i, 3] = np.std(darr)
        sp_up,sp_dn,spr = mkvcal.compExpectHitSteps(tid-lookback,tid,rtn,-rtn,lookfwd)
        # fm[i, 4] = mkvcal.compLimitRtn(tid-lookback, tid, .05, -.05)
        fm[i, 4] = sp_up
        # pdb.set_trace()
        fm[i, 5] = sp_dn

        fm[i, 6] = fm[i, 4]/fm[i, 5]

        print("{} of {} finished".format(i, len(hourids)))

        print('Elapsed time(s): ',time.perf_counter()-st)
        
    np.save(fexconf.getFeatureFile(), fm)
    np.save(fexconf.getLabelFile(), labels)
    print('{} & {} saved'.format(fexconf.getFeatureFile(),fexconf.getLabelFile()))
    plot_labels(fm, labels)

    #compare distribution
    id1 = labels==1
    id2 = labels==2
    for i in range(fm.shape[1]):
        a1 = fm[id1, i]
        a2 = fm[id2, i]
        print("feature {} is same dist: {}".format(i, compare_dist1(a1, a2)))
        compare_cdf(a1, a2, fm[:, i])
        print(ks_2samp(fm[id1, i], fm[id2, i]))

    # check label distribution

    for i in range(fm.shape[1]):
        for j in range(i+1, fm.shape[1]):
            corr, _ =spearmanr(fm[:, i], fm[:, j])
            print('corr {} and {}: {}'.format(i,j,corr))
    #print("label dist. identical with whole? ",compare_dist1(labels[idx], labels))
    plt.show()



