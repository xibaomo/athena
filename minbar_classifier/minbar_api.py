'''
This file provides api functions to be called by athena
'''
import sys
import pandas as pd
import pickle
import yaml
import pdb
from sklearn.preprocessing import *
from prediction import *
from features import *
from labeling import *

CONFIG_FILE=""
HOUR_TIME_ID = []
DATE_STR = ""
TIME_STR = ""
model = None
scaler = MinMaxScaler()
df = pd.DataFrame()
FEXCONF = None

class PredictConfig(object):
    def __init__(self, config_file):
        self.yamlDict = yaml.load(open(config_file))

    def getModelFile(self):
        return self.yamlDict['TRAINING']['MODEL_FILE']

    def getScalerFile(self):
        return self.yamlDict['TRAINING']['SCALER_FILE']

def loadConfig(cf):
    global CONFIG_FILE
    CONFIG_FILE = cf
    pc = PredictConfig(cf)
    global model
    model = pickle.load(open(pc.getModelFile(), 'rb'))
    global scaler
    scaler = pickle.load(open(pc.getScalerFile(), 'rb'))
    global FEXCONF
    FEXCONF = FexConfig(cf)
    print("model and scaler files are loaded")

############ required API for custom py predictor #####################
def init(dates, tms, opens, highs, lows, closes, tkvs):
    print("init() in minbar_api")
    global df, HOUR_TIME_ID
    df = createDataFrame(dates, tms, opens, highs, lows, closes, tkvs)
    for i in range(len(df)):
        tm = pd.to_datetime(df.loc[i, '<TIME>'])
        if tm.minute < 2:
            HOUR_TIME_ID.append(i)

def appendMinbar(dt, tm, op, hp, lp, cp, tkv):
    global df
    df = appendEntryToDataFrame(df, dt, tm, op, hp, lp, cp, tkv)
    t = pd.to_datetime(tm)
    if t.minute < 2:
        HOUR_TIME_ID.append(len(df)-1)

def predict(new_open):
    global df, HOUR_TIME_ID
    tmpdf = appendEntryToDataFrame(df, DATE_STR, TIME_STR, new_open, 0., 0., 0., 0)
    # pdb.set_trace()
    time_id = HOUR_TIME_ID.copy()
    time_id.append(len(tmpdf)-1)
    fm, _, _ = prepare_features(FEXCONF, tmpdf, time_id[-10:])

    fm = scaler.transform(fm)
    y = model.predict(fm)

    if y[-1] == Action.BUY:
        return 1
    if y[-1] == Action.SELL:
        return 2

    return 0 # no action

def test_test(x, y):
    print(x)
    print(y)
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: %s <sym>.csv <fex>.yaml" % sys.argv[0])
        sys.exit(-1)
    tdf = pd.read_csv(sys.argv[1], sep='\t')
    last = tdf.iloc[-1, :]
    tdf = tdf.iloc[:-1, :]
    N = len(tdf)+1-1
    time_id = [N-20, N-16, N-12, N-8, N-4, N]

    loadConfig(sys.argv[2])
    init(tdf['<DATE>'],tdf['<TIME>'],tdf["<OPEN>"],tdf['<HIGH>'],tdf['<LOW>'],tdf['<CLOSE>'],tdf['<TICKVOL>'])
    pred = predict(last['<OPEN>'])
    print(pred)

    print(df.shape)
    appendMinbar(last['<DATE>'],last['<TIME>'],last['<OPEN>'],last['<HIGH>'],last['<LOW>'],last['<CLOSE>'],last['<TICKVOL>'])
    print(df.shape)
