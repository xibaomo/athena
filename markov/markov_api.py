import sys,os
from markov import *
from basics import *
import yaml
import pandas as pd
from scipy.optimize import minimize
import pdb

df = pd.DataFrame()
mkvconf = None
RTN = 0.
Int2Algo = {
    0: "Powell"
}
class MarkovConfig(object):
    def __init__(self,cf):
        self.yamlDict = yaml.load(open(cf))

    def getReturnBounds(self):
        return self.yamlDict['MARKOV']['RETURN_BOUNDS']

    def getOptAlgo(self):
        return self.yamlDict['MARKOV']['OPTIMIZATION']

    def getZeroStateType(self):
        return self.yamlDict['MARKOV']['ZERO_STATE_TYPE']

    def getLookback(self):
        return self.yamlDict['MARKOV']['LOOKBACK']

    def getPosProbThreshold(self):
        return self.yamlDict['MARKOV']['POS_PROB_THRESHOLD']

def loadConfig(cf):
    global mkvconf
    mkvconf = MarkovConfig(cf)

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

############ required API for custom py predictor #####################
def init(dates, tms, opens, highs, lows, closes, tkvs):
    print("init() in markov_api")
    # pdb.set_trace()
    global df
    df = createDataFrame(dates, tms, opens, highs, lows, closes, tkvs)

def appendMinbar(dt, tm, op, hp, lp, cp, tkv):
    global df
    df = appendEntryToDataFrame(df, dt, tm, op, hp, lp, cp, tkv)

def predict(new_time, new_open):
    global df,mkvconf,RTN
    tmpdf = appendEntryToDataFrame(df,"","",new_open,0.,0.,0.,0)
    tarid = len(tmpdf)-1
    lookback = mkvconf.getLookback()
    hist_start = tarid - lookback
    hist_end = tarid

    bnds = mkvconf.getReturnBounds()

    # pdb.set_trace()
    x0 = np.mean(bnds)
    price = new_open
    algo = Int2Algo[mkvconf.getOptAlgo()]

    RTN,prob_buy = max_prob_buy(price,df,hist_start,hist_end,bnds,algo)
    act = 0 # no action
    if prob_buy >= mkvconf.getPosProbThreshold():
        act = 1

    return act
################ END OF PUBLIC API #############
def getReturn():
    global RTN
    return RTN

if __name__ == "__main__":

    csvfile = sys.argv[1]
    ymlfile = sys.argv[2]

    odf = pd.read_csv(csvfile,sep='\t')
    loadConfig(ymlfile)

    tarid = 89443-2
    tdf = odf.iloc[:tarid-1,:]
    init(tdf['<DATE>'],tdf['<TIME>'],tdf["<OPEN>"],tdf['<HIGH>'],tdf['<LOW>'],tdf['<CLOSE>'],tdf['<TICKVOL>'])
    id = tarid-1
    appendMinbar(odf['<DATE>'][id],odf['<TIME>'][id],odf['<OPEN>'][id],odf['<HIGH>'][id],odf['<LOW>'][id],odf['<CLOSE>'][id],odf['<TICKVOL>'][id])

    act = predict("",odf['<OPEN>'].values[tarid])
    rtn = getReturn()
    print("act = {}, rtn = {}".format(act,rtn))