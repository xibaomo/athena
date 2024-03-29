#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 04:09:21 2022

@author: naopc
"""

import sys,os
mkv_path = os.environ['ATHENA_HOME']+'/py_algos/markov'
sys.path.append(mkv_path)
mkv_path = os.environ['ATHENA_HOME']+'/py_algos/mkv_svm'
sys.path.append(mkv_path)
from markov import *
from basics import *
from mkvsvmconf import *
import yaml
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew,kurtosis
from prob_speed import *
from pkl_predictor import *
from feature import *
import pdb

df = pd.DataFrame()
pos_df = pd.DataFrame()
mkvconf = None
RTN = 0.
LAST_DECISION_TIME = None
scaler = None
model = None
oracle = None

def loadConfig(cf):
    global mkvconf
    mkvconf = MkvSvmConfig(cf)

def createDataFrame(dates, tms, opens, highs, lows, closes, tickvols):
    df = pd.DataFrame()
    df[DATE_KEY] = dates
    df[TIME_KEY] = tms
    df[OPEN_KEY] = opens
    df[HIGH_KEY] = highs
    df[LOW_KEY]  = lows
    df[CLOSE_KEY] = closes
    df[TICKVOL_KEY] = tickvols
    df[TIMESTAMP_KEY] = pd.to_datetime(df[DATE_KEY] + " " + df[TIME_KEY])
    return df
def appendEntryToDataFrame(df,timestamp,op,hp,lp,cp,tkv):
    dict = {DATE_KEY: ["-"],
            TIME_KEY: ["-"],
            OPEN_KEY: [round(op,5)],
            HIGH_KEY: [round(hp,5)],
            LOW_KEY : [round(lp,5)],
            CLOSE_KEY: [round(cp,5)],
            TICKVOL_KEY: [tkv],
            TIMESTAMP_KEY: [timestamp]}
    df2 = pd.DataFrame(dict)
    if df2[TIMESTAMP_KEY][0] == df[TIMESTAMP_KEY].values[-1] :
        return df
    df3 = pd.concat([df, df2], ignore_index = True)
    return df3
def createPredictor(gencfg):
    p = None
    tp = gencfg.getPredictorType()
    if tp == 0:
        p = ProbSpeedPredictor(gencfg)
    elif tp == 1:
        p = PklPredictor(gencfg)
    else:
        print('Error: predictor type is wrong: ', tp)
        
    return p
############ required API for custom py predictor #####################
def init(dates, tms, opens, highs, lows, closes, tkvs):
    global oracle,mkvconf
    print("init() in markov_api")
    # pdb.set_trace()
    global df,scaler,model
    df = createDataFrame(dates, tms, opens, highs, lows, closes, tkvs)
    tms = df[DATE_KEY].values[-1] + " " + df[TIME_KEY].values[-1]
    print("Latest open time: ", tms)
    
    oracle = createPredictor(mkvconf)
    
    # mf = mkvconf.getModelFile() 
    # sf = mkvconf.getScalerFile() 
    # model = pickle.load(open(mf,'rb')) 
    # scaler = pickle.load(open(sf,'rb'))

def predict(new_time, new_open):
    global df,mkvconf,RTN, pos_df, LAST_DECISION_TIME,model,scaler,oracle
    
    ts = pd.to_datetime(new_time)
    df = appendEntryToDataFrame(df, ts, new_open, -1, -1, -1, -1)
    nowtime = pd.to_datetime(new_time)

    if LAST_DECISION_TIME is not None:
        dt = nowtime - LAST_DECISION_TIME 
        if dt.total_seconds()/60 < mkvconf.getMinPosInterval():
            return 0
    
    LAST_DECISION_TIME = nowtime
    
    fm = computeFeatures(mkvconf,df)
    
    if np.isnan(fm).any():
        act = 0
    else:
        act = oracle.predict(fm)

    print("action = {}, server time: {}".format(act,new_time))


    registerPos(new_time,new_open,act,fm)
        

    return act

def finalize():
    global pos_df,df,mkvconf
    df.to_csv("all_minbars.csv",index=False)
    pos_df.to_csv("online_decision.csv",index=False)

################ END OF PUBLIC API #############
def computeFeatures(mkvconf,df):
    tarid = len(df)-1
    lookback = mkvconf.getLookback()
    hist_start = tarid - lookback
    hist_end = tarid
    mkvcal = MkvCalEqnSol(df,mkvconf.getNumPartitions())
    
    fm = compMkvFeatures(df,tarid,mkvcal,mkvconf)
   
    return fm
    

def getReturn():
    global RTN,mkvconf
    RTN = mkvconf.getUBReturn()
    return RTN
def registerPos(tm,price,act,fm):
    global pos_df
    dict = {"TIME" : [tm],
            "PRICE" : [price],
            "ACTION" : [act],
            "PROB_BUY": [round(fm[0,0],4)],
            "SPEED": [fm[0,1]],
            "STD" : [fm[0,3]],
            "ACC" : [fm[0,4]]}
    df2 = pd.DataFrame(dict)
    # pdb.set_trace()
    pos_df = pd.concat([pos_df, df2], ignore_index=True)

if __name__ == "__main__":

    csvfile = sys.argv[1]
    ymlfile = sys.argv[2]
    tar_time = sys.argv[3] + ' ' + sys.argv[4]

    odf = pd.read_csv(csvfile,sep='\t')
    loadConfig(ymlfile)
    ts = pd.to_datetime(odf['<DATE>'] + " " + odf['<TIME>'])

    tt = pd.to_datetime(tar_time)

    tarid = ts.index[ts==tt].tolist()[0]    

    tdf = odf.iloc[:tarid,:]
    init(tdf['<DATE>'],tdf['<TIME>'],tdf["<OPEN>"],tdf['<HIGH>'],tdf['<LOW>'],tdf['<CLOSE>'],tdf['<TICKVOL>'])
    id = tarid-1
    
    act = predict("",odf['<OPEN>'].values[tarid])
    rtn = getReturn()
    print(odf['<DATE>'][tarid],odf['<TIME>'][tarid])
    print("act = {}, rtn = {}".format(act,rtn))

    # sys.exit(0)

    






