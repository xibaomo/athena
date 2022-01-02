"""
This file is not intended for training
"""
import pandas as pd
from basics import *

def createDataFrame(dates, tms, opens, highs, lows, closes, tickvols):
    df = pd.DataFrame()
    df[DATE_KEY] = dates
    df[TIME_KEY] = tms
    df[OPEN_KEY] = opens
    df[HIGH_KEY] = highs
    df[LOW_KEY]  = lows
    df[CLOSE_KEY] = closes
    df[TICKVOL_KEY] = tickvols
    return df

def appendEntryToDataFrame(df,dt,tm,op,hp,lp,cp,tkv):
    dict = {DATE_KEY: [dt],
            TIME_KEY: [tm],
            OPEN_KEY: [round(op,5)],
            HIGH_KEY: [round(hp,5)],
            LOW_KEY : [round(lp,5)],
            CLOSE_KEY: [round(cp,5)],
            TICKVOL_KEY: [tkv]}
    df2 = pd.DataFrame(dict)
    df3 = pd.concat([df, df2], ignore_index = True)
    return df3

