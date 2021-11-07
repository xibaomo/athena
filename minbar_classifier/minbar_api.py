'''
This file provides api functions to be called by athena
'''
import pandas as pd
import pickle
from prediction import *

MODEL_FILE = ""
SCALER_FILE = ""

model = None
scaler = None
df = pd.DataFrame()
def init(dates, tms, opens, highs, lows, closes, tkvs):
    df = createDataFrame(dates,tms,opens,highs,lows,closes,tkvs)


def appendMinbar(dt,tm,op,hp,lp,cp,tkv):
    appendEntryToDataFrame(df,dt,tm,op,hp,lp,cp,tkv)

def predict(new_close):
    if scaler is None:
        scaler = pickle.load(open(SCALER_FILE, 'rb'))
    if model is None:
        model = pickle.load(open(MODEL_FILE, 'rb'))

