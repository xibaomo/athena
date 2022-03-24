import sys,os
from markov import *
from basics import *
import yaml
import pandas as pd
from scipy.optimize import minimize
import pdb

df = pd.DataFrame()
pos_df = pd.DataFrame()
mkvconf = None
RTN = 0.

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
    if df2[DATE_KEY][0] == df[DATE_KEY].values[-1] and df2[TIME_KEY][0] == df[TIME_KEY].values[-1]:
        return df
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
    global df,mkvconf,RTN, pos_df
    # pdb.set_trace()
    tmpdf = appendEntryToDataFrame(df,"","",new_open,0.,0.,0.,0)
    tarid = len(tmpdf)-1
    lookback = mkvconf.getLookback()
    hist_start = tarid - lookback
    hist_end = tarid

    price = new_open

    mkvcal = None
    if mkvconf.getProbCalType() == 0:
        mkvcal = MkvProbCalOpenPrice(df, price)
    elif mkvconf.getProbCalType() == 1:
        mkvcal = MkvCalTransMat(df, price, mkvconf.getNumStates())
    elif mkvconf.getProbCalType() == 2:
        mkvcal = MkvCalEqnSol(df,mkvconf.getNumPartitions())
    else:
        pass

    tp = mkvconf.getTPReturn()
    sl = mkvconf.getSLReturn()

    # hist_start=0
    prob_buy = mkvcal.compWinProb(hist_start, hist_end, tp, sl)

    print("buy prob: ", prob_buy)
    # pdb.set_trace()
    # print("sell prob: ", prob_sell)
    # print(prob_buy+prob_sell)
    act = 0 # no action
    prob_thd = mkvconf.getPosProbThreshold()
    prob_pos = prob_buy
    if prob_buy >= prob_thd:
        # pdb.set_trace()
        act = 1
    if mkvconf.isBuyOnly() == 0 and 1-prob_buy >= prob_thd:
        prob_pos = 1-prob_buy
        act = 2
    RTN = tp
    if act!=0:
        pos_df = registerPos(pos_df,new_time,act,tp,prob_buy,1-prob_buy,1.0,0)

    return act
def finalize():
    global pos_df,df,mkvconf
    df.to_csv("all_minbars.csv",index=False)
    pos_df.to_csv("online_decision.csv",index=False)

    if mkvconf.isBuyOnly() == 1:
        print("Ave prob of positions: ", np.mean(pos_df['PROB_BUY'].values))

    pass

################ END OF PUBLIC API #############
def getReturn():
    global RTN
    return RTN
def registerPos(df,tm,act,rtn,prob_buy,prob_sell,prob_sum,steps):
    global pos_df
    dict = {"TIME" : [tm],
            "ACTION" : [act],
            "TP_RETURN": [round(rtn,4)],
            "PROB_BUY": [round(prob_buy,4)],
            "PROB_SELL": [round(prob_sell,4)],
            "PROB_SUM": [round(prob_sum,4)],
            "EXP_STEPS": [round(steps)]}
    df2 = pd.DataFrame(dict)
    # pdb.set_trace()
    df3 = pd.concat([df, df2], ignore_index=True)
    return df3

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    # Log.setlogLevel(LOG_INFO)
    csvfile = sys.argv[1]
    ymlfile = sys.argv[2]
    tar_time = sys.argv[3] + ' ' + sys.argv[4]

    odf = pd.read_csv(csvfile,sep='\t')
    loadConfig(ymlfile)
    ts = pd.to_datetime(odf['<DATE>'] + " " + odf['<TIME>'])

    tt = pd.to_datetime(tar_time)

    tarid = ts.index[ts==tt].tolist()[0]    

    tdf = odf.iloc[:tarid-1,:]
    init(tdf['<DATE>'],tdf['<TIME>'],tdf["<OPEN>"],tdf['<HIGH>'],tdf['<LOW>'],tdf['<CLOSE>'],tdf['<TICKVOL>'])
    id = tarid-1
    appendMinbar(odf['<DATE>'][id],odf['<TIME>'][id],odf['<OPEN>'][id],odf['<HIGH>'][id],odf['<LOW>'][id],odf['<CLOSE>'][id],odf['<TICKVOL>'][id])

    act = predict("",odf['<OPEN>'].values[tarid])
    rtn = getReturn()
    print(odf['<DATE>'][tarid],odf['<TIME>'][tarid])
    print("act = {}, rtn = {}".format(act,rtn))

    from scipy.stats import skew,kurtosis
    # global mkvconf
    lk = mkvconf.getLookback()
    p = odf['<OPEN>'].values[tarid-lk:tarid+1]
    r = np.diff(np.log(p))
    sk =skew(r)
    kt = kurtosis(r)-3
    from scipy.optimize import fsolve
    fs = lambda x: 2*(1-x**6) - sk*(x**4+1)**(3/2)
    fk = lambda x: 6*(1+x**8) - kt*(1+x**4)**2
    k = fsolve(fs,[0.5,1,5,10])
    print(k)









