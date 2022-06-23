import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import IntEnum
from logger import *
from basics import *
import joblib
import pdb

import sys,os
athena_path= "%s/api/api_athena/pyapi" % os.environ["ATHENA_HOME"]
sys.path.append(athena_path)
import athena as atn

class Action(IntEnum):
    BUY = 0,
    SELL = 1,
    NO_ACTION = 2

def getTimeStamps(df):
    dtm = df[DATE_KEY]+" "+df[TIME_KEY]
    tm = pd.to_datetime(dtm.values)
    return tm

def inst_change_label(df):
    '''
    :param rx: return sequence
    :param thd: threshold
    :return: labels at each return
    '''
    tm = getTimeStamps(df)
    time_id = (tm.minute==0) & (tm.second==0)
    prices = df[OPEN_KEY].values[time_id]
    rx = np.diff(np.log(prices))

    thd = 0.5*np.std(rx)
    y = np.ones(len(rx))*np.nan
    for i in range(len(rx)):
        if rx[i] > thd:
            y[i] = Action.BUY
        elif rx[i] < -thd:
            y[i] = Action.SELL
        else:
            y[i] = Action.NO_ACTION

    p = plt.hist(y)[0]
    print("label dist.: ", p)
    print("thd: ",thd)
    return y,time_id

def tailorTargetID(tm,target_id,pos_life):
    """
    remove tailing time id whose life is shorter than 1 week
    """
    t_end = tm[target_id[-1]]
    idx = len(target_id)-2 #index to target_id
    while 1:
        tid = target_id[idx] # index to tm
        t_cur = tm[tid]
        dt = (t_end-t_cur).total_seconds()
        if dt >= pos_life:
            break
        idx-=1
    tailored_idx = target_id[:idx]
    return tailored_idx

def findLabel(idx,time_id,tm,df,thd_ret,pos_life):
    tid = time_id[idx]
    t0 = tm[tid]  # position time
    id = tid
    p0 = df[OPEN_KEY][id]
    label = None
    while id < len(df):
        t = tm[id]
        if (t - t0).seconds > pos_life:
            break
        # ret+=rx[id]
        ret_high = df[HIGH_KEY][id] / p0 - 1
        ret_low = df[LOW_KEY][id] / p0 - 1
        if ret_high >= thd_ret and ret_low <= -thd_ret:
            Log(LOG_DEBUG) << "Run into a big bar " + df[DATE_KEY][id] + " " + df[TIME_KEY][id]
            label = Action.NO_ACTION
            break
        if ret_high >= thd_ret:
            label = Action.BUY
            break
        if ret_low <= -thd_ret:
            label = Action.SELL
            break
        id+=1
    return label


def later_change_label(df,thd_ret,ret_ratio, pos_life, bar_min=15):
    Log(LOG_INFO) << "Labeling with return threshold: %f, position lifetime: %d days" % (thd_ret,pos_life/3600/24)
    tm = getTimeStamps(df)

    tmp = (tm.minute == 0) & (tm.second == 0)  & (tm.hour != 0)# on hour sharp except 00:00, [true,false,...]
    time_id = [id for id in range(len(tmp)) if tmp[id]] #index to entire df
    time_id = time_id[120:] #make some room for lookback

    time_id = tailorTargetID(tm,time_id,pos_life) # tailor short-life positions
    labels = np.ones(len(time_id))*int(Action.NO_ACTION)
    # labels_ref = np.ones(len(time_id)) * int(Action.NO_ACTION)
    prices = df[OPEN_KEY].values

    time_id = np.array(time_id,dtype=np.int32)

    max_stride = int(pos_life/bar_min/60)
    Log(LOG_INFO) << "Max stride %d" % max_stride
    labels_aux,durations = atn.minbar_label(df[OPEN_KEY].values,df[HIGH_KEY].values,df[LOW_KEY].values,df[CLOSE_KEY].values,df[SPREAD_KEY].values,
                     time_id,thd_ret,ret_ratio,max_stride)

    for i in range(len(labels)):
        if labels_aux[i] == -1:
            labels[i] = Action.SELL
        elif labels_aux[i] == 1:
            labels[i] = Action.BUY

    Log(LOG_INFO) << "Labeling done. All data size: %d"%len(time_id)
    Log(LOG_INFO) << "label dist.: buy: %d, sell: %d, noact: %d "%((labels==Action.BUY).sum(),\
                                                                  (labels==Action.SELL).sum(),
                                                                  (labels==Action.NO_ACTION).sum())

    # add column of ending datetime
    end_time = []
    end_high = []
    end_low  = []
    dts = pd.to_datetime(df.loc[:,DATE_KEY] + " " + df.loc[:,TIME_KEY])
    for i in range(len(labels)):
        tid = time_id[i]
        eid = tid + durations[i]
        end_time.append(dts[eid])
        end_high.append(df.loc[eid,HIGH_KEY])
        end_low.append(df.loc[eid,LOW_KEY])

    return labels,time_id,end_time,end_high,end_low