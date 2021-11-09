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

class PredictConfig(object):
    def __init__(self,config_file):
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
    model = pickle.load(open(pc.getModelFile(),'rb'))
    global scaler
    scaler= pickle.load(open(pc.getScalerFile(),'rb'))

def setHourTimeID(time_id):
    global HOUR_TIME_ID
    HOUR_TIME_ID= time_id

############ required API for custom py predictor #####################
def init(dates, tms, opens, highs, lows, closes, tkvs):
    global df
    df = createDataFrame(dates,tms,opens,highs,lows,closes,tkvs)

def appendMinbar(dt,tm,op,hp,lp,cp,tkv):
    global df
    df = appendEntryToDataFrame(df,dt,tm,op,hp,lp,cp,tkv)

def predict(new_open):
    tmpdf = df
    tmpdf = appendEntryToDataFrame(tmpdf,DATE_STR,TIME_STR,new_open,0.,0.,0.,0)
    # pdb.set_trace()
    fm,_,_ = prepare_features(CONFIG_FILE, tmpdf, HOUR_TIME_ID)

    fm = scaler.transform(fm)
    y = model.predict(fm)

    if y[-1] == Action.BUY:
        return 1
    if y[-1] == Action.SELL:
        return 2

    return 0 # no action

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: %s <sym>.csv <fex>.yaml" % sys.argv[0])
        sys.exit(-1)
    tdf = pd.read_csv(sys.argv[1],sep='\t')
    last = tdf.iloc[-1,:]
    tdf = tdf.iloc[:-1,:]
    N = len(tdf)+1-1
    time_id = [N-20,N-16,N-12,N-8,N-4,N]

    loadConfig(sys.argv[2])
    init(tdf['<DATE>'],tdf['<TIME>'],tdf["<OPEN>"],tdf['<HIGH>'],tdf['<LOW>'],tdf['<CLOSE>'],tdf['<TICKVOL>'])
    setHourTimeID(time_id)
    pred = predict(last['<OPEN>'])
    print(pred)

    print(df.shape)
    appendMinbar(last['<DATE>'],last['<TIME>'],last['<OPEN>'],last['<HIGH>'],last['<LOW>'],last['<CLOSE>'],last['<TICKVOL>'])
    print(df.shape)
