#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:40:32 2022

@author: naopc
"""

import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mkv_path = os.environ['ATHENA_HOME']+'/py_algos/mkv_svm'
sys.path.append(mkv_path)
from conf import *

TM_KEY = 'DATETIME'

def isSharpHour(tm):
    if tm.minute == 0 and tm.second==0:
        return True
    
    return False

def find_range(df,stm,ndays):
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
        return -1,-1
    
    eid = sid+1
 
    while eid < len(df):
        dt = etm - df[TM_KEY][eid]
        if dt.total_seconds() <0:
            break
        eid+=1
        
    return sid,eid
    

def labelHours(df,sid,eid,rtn):
    labels = []
    tid= []
    for i in range(sid,eid):
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
                if dt.total_seconds() >= 3600*24*7:
                    if p0 > rh:
                        lb = 2
                    if p0 < rl:
                        lb = 1
                    
                    break
                pass 
            idx+=1
        labels.append(lb)
        
    return np.array(labels),np.array(tid)
        

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} <csv> <mkv.yaml> <date> <time> <ndays>".format(sys.argv[0]))
        sys.exit(0)
        
    csvfile = sys.argv[1]
    ymlfile = sys.argv[2]
    start_time = pd.to_datetime(sys.argv[3] + " " + sys.argv[4])
    ndays = sys.argv[5]
    
    print(start_time)
    
    df = pd.read_csv(csvfile,sep='\t')
    mkvconf = MarkovConfig(ymlfile)
    
    sid,eid = find_range(df, start_time, ndays)
    if sid < 0:
        print("Target datetime not found: {}",start_time)
        sys.exit(0)
        
    rtn = mkvconf.getUBReturn()
        
    labels,hourids = labelHours(df, sid, eid, rtn)   
    
    ##### markov
    mkvcal = MkvCalEqnSol(df,mkvconf.getNumPartitions())
    
    lookback = mkvconf.getLookback()
    fm = np.zeros((len(labels),2))
    for i in range(len(hourids)):
        hist_end = hourids[i] + 1
        hist_start = hist_end - lookback
        prop,sp = mkvcal.compWinProb(hist_start,hist_end,rtn,-rtn)
        fm[i,0] = prop
        fm[i,1] = rtn/sp 
        
    print('features done')
    np.save('fm.npy',fm)
    np.save('labels.npy',labels)
    
    print('fm.npy & labels.npy saved')
    

        
        
        
        
        