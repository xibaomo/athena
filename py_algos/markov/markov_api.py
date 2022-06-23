import sys,os
from markov import *
from basics import *
import yaml
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew,kurtosis
import pdb

df = pd.DataFrame()
pos_df = pd.DataFrame()
mkvconf = None
RTN = 0.
all_spds = []
WAVE_START_ID = -1
LAST_POS_TIME = None

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
    tms = df[DATE_KEY].values[-1] + " " + df[TIME_KEY].values[-1]
    print("Latest open time: ", tms)

def appendMinbar(dt, tm, op, hp, lp, cp, tkv):
    global df
    df = appendEntryToDataFrame(df, dt, tm, op, hp, lp, cp, tkv)

def predict(new_time, new_open):
    global df,mkvconf,RTN, pos_df, all_spds, WAVE_START_ID,LAST_POS_TIME
    nowtime = pd.to_datetime(new_time)
    # pdb.set_trace()
    # tmpdf = appendEntryToDataFrame(df,"","",new_open,0.,0.,0.,0)
    tarid = len(df)-1
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
    elif mkvconf.getProbCalType() == 3:
        mkvcal = FirstHitProbCal(df,mkvconf.getNumPartitions())
    else:
        pass

    tp = mkvconf.getUBReturn()
    sl = mkvconf.getLBReturn()

    prob_buy,steps = mkvcal.compWinProb(hist_start,hist_end,tp,sl)
    spd = tp/steps
    all_spds.append(spd)
    npts = mkvconf.getSpeedLookback()
    grad = compSpeedGrad(all_spds,npts,3)
    offscaler = mkvconf.getOffPeakSpeedScaler()
    act = 0 # no action
    ub_prob = mkvconf.getBuyProb()
    lb_prob = mkvconf.getSellProb()
    spd_thd = mkvconf.getOpenPosSpeed()
    if WAVE_START_ID < 0:
        ave_spd = 0.01/1440
        if len(all_spds) > npts:
            ave_spd = np.mean(all_spds[-npts:])
        print("{}-ave speed: {} ".format(npts,ave_spd))
        if spd > ave_spd and grad > 0:
            WAVE_START_ID = len(all_spds) - 1
            if prob_buy>= ub_prob:
                act = 1 # long
            elif prob_buy <= lb_prob:
                act = 2 # short
            else:
                pass

    else:
        s = all_spds[WAVE_START_ID:]
        high = np.max(s)

        if spd / high <= offscaler and grad < 0:
            # pdb.set_trace()
            print("start speed: {}, peak speed: {}".format(all_spds[WAVE_START_ID],high))
            act = 30
            WAVE_START_ID = -1
        else:
            dt = nowtime - LAST_POS_TIME
            if dt.total_seconds()/60 < mkvconf.getPosIntervalMin():
                act = 0
            else:
                if prob_buy>= ub_prob:
                    act = 1 # long
                elif prob_buy <= lb_prob:
                    act = 2 # short
                else:
                    pass

    print("prop = {}, spd = {}, grad = {}".format(prob_buy,spd,grad))

    print("action = ", act)

    if act ==1 or act==2:
        # pdb.set_trace()
        registerPos(new_time,act,prob_buy,spd)
        LAST_POS_TIME = nowtime

    return act

def finalize():
    global pos_df,df,mkvconf
    df.to_csv("all_minbars.csv",index=False)
    pos_df.to_csv("online_decision.csv",index=False)

################ END OF PUBLIC API #############
def compSpeedGrad(spds,n_pts,order):
    # pdb.set_trace()
    if len(spds) < n_pts:
        return 0
    allspeeds  = np.array(spds)
    y = allspeeds[-n_pts:]
    x = np.array([i for i in range(n_pts)])
    p = np.polyfit(x,y,order)
    x0 = n_pts-1
    dx = 1e-4
    g = (np.polyval(p,x0+dx) - np.polyval(p,x0-dx))/2./dx
    return g

def compLongSig(mkvcal,mkvconf,tarid,ub_rtn,lb_rtn):
    llk = mkvconf.getLongLookback()
    lpb,lps = mkvcal.comp1stHitProb(tarid-llk,tarid,ub_rtn,lb_rtn,mkvconf.getSteps())
    sig = lpb/(lpb+lps)
    return sig

def getReturn():
    global RTN
    return RTN
def registerPos(tm,act,prob_buy,spd):
    global pos_df
    dict = {"TIME" : [tm],
            "ACTION" : [act],
            "PROB_BUY": [round(prob_buy,4)],
            "SPEED": [spd]}
    df2 = pd.DataFrame(dict)
    # pdb.set_trace()
    pos_df = pd.concat([pos_df, df2], ignore_index=True)

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    # Log.setlogLevel(LOG_INFO)
    csvfile = sys.argv[1]
    ymlfile = sys.argv[2]
    tar_time = sys.argv[3] + ' ' + sys.argv[4]

    odf = pd.read_csv(csvfile,sep=',')
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

    # sys.exit(0)

    from scipy.stats import skew,kurtosis
    # global mkvconf
    lk = mkvconf.getLookback()
    p = odf['<OPEN>'].values[tarid-lk:tarid+1]
    r = np.diff(np.log(p))
    p1 = odf['<OPEN>'].values[tarid:tarid+lk]
    r1 = np.diff(np.log(p1))
    from scipy.stats import ks_2samp
    res = ks_2samp(r1,r)

    # pdb.set_trace()
    tp = mkvconf.getUBReturn()
    mkvcal = MkvCalEqnSol(odf,mkvconf.getNumPartitions())
    # mkvcal = FirstHitProbCal(odf, mkvconf.getNumPartitions())

    ftu = int(sys.argv[-1])
    ftu = min(len(odf)-tarid,ftu)
    pbs = []
    pss = []
    pc = []
    p0=odf['<OPEN>'][tarid]

    lsig = []
    kh=-1
    props=[]
    spds = []
    grads = []
    lookback = mkvconf.getLookback()
    for i in range(00, ftu):
        tm = odf['<DATE>'][tarid+i] + " " + odf['<TIME>'][tarid+i]
        tm = pd.to_datetime(tm)
        # if tm == tartm:
        #     pdb.set_trace()
        if tm.second > 0 or tm.minute > 0:
            continue
        hist_end = tarid+i
        if hist_end >= len(odf):
            break
        kh+=1
        hist_start = hist_end - lookback
        prop,sp = mkvcal.compWinProb(hist_start,hist_end,tp,-tp)
        props.append(prop)
        # _,sp = mkvcal.compWinProb(hist_end-60,hist_end,tp,-tp)
        vsp = tp/sp
        spds.append(vsp)

        g = compSpeedGrad(spds,mkvconf.getSpeedLookback(),3)
        grads.append(g)

        # prob_buy,prob_sell = mkvcal.comp1stHitProb(hist_start,hist_end,tp,-tp,mkvconf.getSteps())
        # pbs.append(prob_buy)
        # pss.append(prob_sell)
        # p = odf['<OPEN>'].values[hist_start:hist_end+1]
        # rtn = np.diff(np.log(p))
        # sw = skew(rtn)
        # sk.append(sw)
        pc.append(odf['<OPEN>'][hist_end]/p0-1)
        # lpb,lps = mkvcal.comp1stHitProb(hist_end-mkvconf.getLongLookback(),hist_end,tp,-tp,mkvconf.getSteps())
        # lsg = lpb/(lpb+lps)
        # lsig.append(lsg)
        # pc.append(odf['<OPEN>'][hist_end])

        print(odf['<DATE>'][hist_end],odf['<TIME>'][hist_end],prop,sp,kh)


    # pc = odf['<OPEN>'].values[tarid-1000:tarid+ftu]
    # pbs = np.array(pbs)
    # pss = np.array(pss)
    fig,(ax1,ax2) = plt.subplots(2,1)
    ax11 = ax1.twinx()
    ax11.plot(spds,'y.-')
    ax1.plot(pc,'r.-')

    ax22 = ax2.twinx()
    ax22.plot(props,'b.-')
    ax2.plot(pc,'r.-')


    plt.title('lookback: ' + str(lookback) + 'minbar')
    plt.show()








