#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:48:26 2022

@author: naopc
"""
import numpy as np
import pdb

all_speed=[]

def getNumFeatures():
    return 5

def compMkvFeatures(df,tid,mkvcal,mkvconf):
    global all_speed
    lookback = mkvconf.getLookback()
    rtn = mkvconf.getUBReturn()
    fm = np.zeros((1,getNumFeatures()))
    hist_end = tid 
    hist_start = hist_end - lookback
    prop,sp = mkvcal.compWinProb(hist_start,hist_end,rtn,-rtn)
    fm[0,0] = prop
    fm[0,1] = rtn/sp
    if prop < 0.5:
        fm[0,1] = -fm[0,1]

    rts = mkvcal.getRtn()
    fm[0,2] = sum(rts)
    fm[0,3] = np.std(rts)
    all_speed.append(fm[0,1])
    fm[0,4] = compAcceleration(all_speed)
    return fm

def compAcceleration(spd,lookback=12):
    # pdb.set_trace()
    if len(spd) < lookback+1:
        return np.nan
    x=[x for x in range(lookback+1)]
    y = spd[-(lookback+1):]
    
    # pdb.set_trace()
    c = np.polyfit(x,y,1)
    
    return c[0]