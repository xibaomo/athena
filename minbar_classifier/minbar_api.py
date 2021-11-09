'''
This file provides api functions to be called by athena
'''
import pandas as pd
import pickle
import yaml
from sklearn.preprocessing import *
from prediction import *
from features import *
from labeling import *

CONFIG_FILE=""
HOUR_TIME_ID = []
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

pc = None
def loadConfig(cf):
    CONFIG_FILE = cf
    pc = PredictConfig(cf)
    model = pickle.load(open(pc.getModelFile(),'rb'))
    scaler = pickle.load(open(pc.getScalerFile(),'rb'))

def setHourTimeID(time_id):
    HOUR_TIME_ID = time_id
############ required API for custom py predictor #####################
def init(dates, tms, opens, highs, lows, closes, tkvs):
    df = createDataFrame(dates,tms,opens,highs,lows,closes,tkvs)

def appendMinbar(dt,tm,op,hp,lp,cp,tkv):
    appendEntryToDataFrame(df,dt,tm,op,hp,lp,cp,tkv)

def predict(date_str, time_str, new_open):
    tmpdf = df
    appendEntryToDataFrame(tmpdf,date_str,time_str,new_open,0.,0.,0.,0)

    fm,_,_ = prepare_features(CONFIG_FILE, tmpdf, HOUR_TIME_ID)

    fm = scaler.transform(fm)
    y = model.predict(fm)

    if y[-1] == Action.BUY:
        return 1
    if y[-1] == Action.SELL:
        return 2

    return 0 # no action

