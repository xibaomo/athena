#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:48:26 2022

@author: naopc
"""
import numpy as np

def compMkvFeatures(df,tid,mkvcal,mkvconf):
    lookback = mkvconf.getLookback()
    rtn = mkvconf.getUBReturn()
    fm = np.zeros((1,4))
    hist_end = tid + 1
    hist_start = hist_end - lookback
    prop,sp = mkvcal.compWinProb(hist_start,hist_end,rtn,-rtn)
    fm[0,0] = prop
    fm[0,1] = rtn/sp

    ops = df['<OPEN>'].values[hist_start:hist_end]
    rts = np.diff(np.log(ops))
    fm[0,2] = sum(rts)
    fm[0,3] = np.std(rts)
    
    return fm