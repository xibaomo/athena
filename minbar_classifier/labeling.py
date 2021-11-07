import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import IntEnum
from logger import *
from basics import *
import pdb

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

def tailorTargetID(tm,target_id,POS_LIFE):
    """
    remove tailing time id whose life is shorter than 1 week
    """
    t_end = tm[target_id[-1]]
    idx = len(target_id)-2 #index to target_id
    while 1:
        tid = target_id[idx] # index to tm
        t_cur = tm[tid]
        dt = (t_end-t_cur).total_seconds()
        if dt >= POS_LIFE:
            break
        idx-=1
    tailored_idx = target_id[:idx]
    return tailored_idx


def later_change_label(df,THD_RET,POS_LIFE):
    Log(LOG_INFO) << "Labeling with return threshold: %f, position lifetime: %d days" % (THD_RET,POS_LIFE/3600/24)
    tm = getTimeStamps(df)

    tmp = (tm.minute == 0) & (tm.second == 0) # on hour sharp, [true,false,...]
    time_id = [id for id in range(len(tmp)) if tmp[id]] #index to entire df
    time_id = time_id[120:] #make some room for lookback

    time_id = tailorTargetID(tm,time_id,POS_LIFE) # tailor short-life positions
    labels = np.ones(len(time_id))*int(Action.NO_ACTION)

    prices = df[OPEN_KEY].values
    rx = np.diff(np.log(prices))
    for i in range(len(time_id)):
        tid = time_id[i] #index to tm
        t0 = tm[tid]     # position time
        ret = 0.
        id = tid
        while id < len(rx):
            t = tm[id]
            if (t-t0).seconds > POS_LIFE:
                break
            ret+=rx[id]
            if (ret > THD_RET):
                labels[i] = Action.BUY
                break
            if (ret < -THD_RET):
                labels[i] = Action.SELL
                break
            id+=1

    Log(LOG_INFO) << "All data size: %d"%len(time_id)
    Log(LOG_INFO) << "label dist.: buy: %d, sell: %d, noact: %d "%((labels==Action.BUY).sum(),\
                                                                  (labels==Action.SELL).sum(),
                                                                  (labels==Action.NO_ACTION).sum())

    return labels,time_id