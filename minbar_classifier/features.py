import pdb

import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from logger import *
from basics import *
import yaml

from statsmodels.tsa.arima.model import ARIMA

def __prepare_features(df, time_id, labels, test_size):
    lookback = 3
    prices = df[OPEN_KEY].values[time_id]
    rx = np.diff(np.log(prices))
    tv = df["<TICKVOL>"].values[time_id]
    tm = pd.to_datetime(df[TIME_KEY].values[time_id])
    tv = tv[:-1]
    tm = tm[:-1]

    nsample = len(rx) - lookback
    nf = 5
    fm = np.zeros((nsample, nf))
    label = np.ones(nsample)*np.nan
    for i in range(nsample):
        past = rx[i:i+lookback]
        tvp  = tv[i:i+lookback]
        fm[i, 0] = np.mean(past)
        fm[i, 1] = np.std(past)
        fm[i, 2] = np.mean(tvp)
        fm[i, 3] = np.std(tvp)
        #fm[i, 5] = fm[i, 3]/fm[i, 4]
        fm[i, 4] = tm[i+lookback].hour
        #fm[i, 5] = fm[i, 0]/fm[i, 1]
        label[i] = labels[i+lookback]

    x_train = fm[:-test_size, :]
    y_train = label[:-test_size]
    x_test = fm[-test_size:, :]
    y_test = label[-test_size:]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, y_train, x_test, y_test
BASICS_KEY      = "BASICS"
KAMA_KEY        = "KAMA"
VOLATILITY_KEY  = "VOLATILITY"
WMA_KEY         = "WMA"
BBANDS_KEY      = "BBANDS"
MACD_KEY        = "MACD"

LOOKBACK_KEY = "LOOKBACK"
LONG_LOOKBACK_KEY = "LONG_LOOKBACK"
class FexConfig(object):
    def __init__(self, cf):
        self.__yamlDict = yaml.load(open(cf))
        self.fexDict = self.__yamlDict['FEATURES']

    def getLookback(self, key):
        return self.fexDict[key][LOOKBACK_KEY]
    def getLongLookback(self, key):
        return self.fexDict[key][LONG_LOOKBACK_KEY]
    def getFeatureParam(self,feature,param):
        return self.fexDict[feature][param]

    def isFeatureEnabled(self, key):
        if not key in self.fexDict:
            return False
        if self.fexDict[key][LOOKBACK_KEY] < 0:
            return False
        return True

def volatility(df, time_id, slk = 3,llk=6):
    Log(LOG_INFO) << "Computing volatility with %d, %d" % (slk,llk)
    s = np.zeros(len(df))
    for i in range(len(df)-1):
        if df[OPEN_KEY][i] > df[CLOSE_KEY][i]:
            s[i] = df[HIGH_KEY][i]/df[LOW_KEY][i]
        else:
            s[i] = df[LOW_KEY][i] / df[HIGH_KEY][i]
    df[STD_KEY] = s
    ft = np.zeros((len(time_id), 4))
    for i in range(len(time_id)):
        tid = time_id[i]
        past_sd = df[STD_KEY][tid-slk:tid].values
        ft[i, 0] = np.mean(past_sd)
        ft[i, 1] = np.std(past_sd)
        past_sd_l = df[STD_KEY][tid-llk:tid].values
        ft[i,2] = np.mean(past_sd_l)
        ft[i,3] = np.std(past_sd_l)
    return ft
def dema(df, time_id, slk = 6, llk = 12):
    Log(LOG_INFO) << "Computing dema with %d, %d" % (slk, llk)
    prices = df[OPEN_KEY].values
    ma_short = talib.DEMA(prices, slk)
    ma_long  = talib.DEMA(prices, llk)
    d = prices[time_id] - ma_short[time_id]
    dma = ma_short[time_id] - ma_long[time_id]

    return np.hstack((d.reshape(-1, 1), dma.reshape(-1, 1)))

def kama(df, time_id, slk = 3, llk = 9):
    Log(LOG_INFO) << "Computing kama with %d, %d" %(slk, llk)
    prices = df[OPEN_KEY].values
    kma_s = talib.KAMA(prices, slk)
    kma_l = talib.KAMA(prices, llk)
    dif_s = prices[time_id] - kma_s[time_id]
    dif_l = prices[time_id] - kma_l[time_id]
    dkma = kma_s[time_id] - kma_l[time_id]
    return np.hstack((dif_s.reshape(-1, 1), dif_l.reshape(-1,1), dkma.reshape(-1, 1)))

def wma(df, time_id, slk = 3, llk = 24):
    Log(LOG_INFO) << "Computing wma with %d, %d" % (slk, llk)
    prices = df[OPEN_KEY].values
    kma_s = talib.WMA(prices, slk)
    kma_l = talib.WMA(prices, llk)
    dif_s = prices[time_id] - kma_s[time_id]
    dif_l = prices[time_id] - kma_l[time_id]
    dwma = kma_s[time_id] - kma_l[time_id]
    return np.hstack((dif_s.reshape(-1, 1), dif_l.reshape(-1,1),dwma.reshape(-1, 1)))

def bbands(df, time_id, lk = 48):
    Log(LOG_INFO) << "Computing bbands with %d" % lk
    prices = df[OPEN_KEY].values
    ndev = 0.618*2
    up, mid, low = talib.BBANDS(prices, lk, ndev, ndev)
    labels = np.zeros(len(time_id))
    for i in range(len(time_id)):
        tid = time_id[i]
        pc = prices[tid]
        if pc > up[tid]:
            labels[i] = 1
        if pc < low[tid]:
            labels[i] = -1
    Log(LOG_INFO) << "bbands labels dist. up %.2f, down %.2f" % ((labels==1).sum()*1./len(labels), (labels==-1).sum()*1./len(labels))
    return labels.reshape(-1, 1)

def macd(df, time_id, slk = 48, llk = 96, sp = 9):
    Log(LOG_INFO) << "Computing macd with %d, %d, %d" %(slk, llk, sp)
    macd, macdsignal, macdhist = talib.MACD(df[CLOSE_KEY].values, slk, llk, sp)
    macd = macd[time_id]
    macdsignal = macdsignal[time_id]
    macdhist = macdhist[time_id]
    for i in range(len(macd)):
        if macd[i] > 0:
            macd[i] = 1
        elif macd[i] < 0:
            macd[i] = -1
        else:
            macd[i] = 0
    return np.hstack((macd.reshape(-1, 1), macdsignal.reshape(-1, 1), macdhist.reshape(-1, 1)))
def momentum(df,time_id,lookback):
    Log(LOG_INFO) << "Computing momentum with looback: %d" % lookback
    mmt_open=[]
    mmt_high=[]
    mmt_mid =[]
    for tid in time_id:
        oldp = df[OPEN_KEY][tid-lookback]
        newp = df[OPEN_KEY][tid]
        rt = newp/oldp - 1
        mmt_open.append(rt)

        oldp = df[HIGH_KEY][tid - lookback]
        newp = df[HIGH_KEY][tid-1]
        rt = newp/oldp - 1
        mmt_high.append(rt)

        oldp = df[MID_KEY][tid - lookback]
        newp = df[OPEN_KEY][tid]
        rt = newp / oldp - 1
        mmt_mid.append(rt)

    mmt_open = np.array(mmt_open)
    mmt_high = np.array(mmt_high)
    mmt_mid  = np.array(mmt_mid)

    return np.hstack((mmt_open.reshape(-1,1),mmt_high.reshape(-1,1),mmt_mid.reshape(-1,1)))

def cci(df,time_id,slk,llk):
    Log(LOG_INFO) << "Computing CCI with lookback: %d, %d" % (slk,llk)
    # pdb.set_trace()
    cci_s = talib.CCI(df[HIGH_KEY].values, df[LOW_KEY].values, df[CLOSE_KEY].values,slk)
    cci_s = cci_s[time_id-1]

    cci_l = talib.CCI(df[HIGH_KEY].values, df[LOW_KEY].values, df[CLOSE_KEY].values, llk)
    cci_l = cci_l[time_id - 1]
    return np.hstack((cci_s.reshape(-1,1),cci_l.reshape(-1,1)))

def rsi(df,time_id,slk,llk):
    Log(LOG_INFO) << "Computing RSI wtih lookback: %d,%d" % (slk,llk)
    rsi_s = talib.RSI(df[OPEN_KEY].values,slk)[time_id]
    rsi_l = talib.RSI(df[OPEN_KEY].values,llk)[time_id]
    rsi = rsi_s - rsi_l
    return np.hstack((rsi_s.reshape(-1,1),rsi_l.reshape(-1,1),rsi.reshape(-1,1)))

def rsv(df,time_id,lookback):
    Log(LOG_INFO) << "Computing RSV with lookback: %d" % lookback
    rsv = []
    for tid in time_id:
        high = np.max(df[HIGH_KEY][tid-lookback:tid])
        low  = np.min(df[LOW_KEY][tid-lookback:tid])
        val = (df[OPEN_KEY][tid]-low)/(high-low)
        rsv.append(val)
    rsv = np.array(rsv)
    return rsv.reshape(-1,1)

def slope(df,time_id,slk,llk):
    Log(LOG_INFO) << "Computing SLOPE with lookback: %d,%d" % (slk,llk)
    xs = range(slk)
    xl = range(llk)
    slp_s = []
    slp_l = []
    slp_d = []
    for tid in time_id:
        ys = df[MID_KEY][tid-slk:tid]
        yl = df[MID_KEY][tid-llk:tid]
        ss = np.polyfit(xs,ys,1)[0]
        sl = np.polyfit(xl,yl,1)[0]
        slp_s.append(ss)
        slp_l.append(sl)
        slp_d.append(ss/sl)
    slp_s = np.array(slp_s)
    slp_l = np.array(slp_l)
    slp_d = np.array(slp_d)
    return np.hstack((slp_s.reshape(-1,1),slp_l.reshape(-1,1),slp_d.reshape(-1,1)))

def arima(df,time_id,lookback, p,d,q):
    Log(LOG_INFO) << "Computing arima(%d,%d,%d) with lookback: %d " % (p,d,q,lookback)
    pd=[]
    for tid in time_id:
        # pdb.set_trace()
        series = np.log(df[OPEN_KEY][tid-lookback:tid].values)
        model = ARIMA(series,order=(p,d,q))
        model_fit = model.fit(method_kwargs={"warn_convergence": False})
        output = model_fit.forecast()
        p0 = np.log(df[OPEN_KEY][tid])
        err = (output[0]-p0)/p0
        pd.append(err)

    pd = np.array(pd)
    return pd.reshape(-1,1)

def tsfresh(df,time_id,lookback):
    from tsfresh import extract_features
    from tsfresh import select_features
    from tsfresh.utilities.dataframe_functions import impute
    Log(LOG_INFO) << "Computing TSFRESH with lookback: %d" % lookback
    hid_key = "HOUR_ID"
    ts_key = "TIME_STAMP"
    dt_key = "DURATION"
    dff = pd.DataFrame()
    id=0
    for tid in time_id:
        tmp = df.iloc[tid-lookback:tid,:].reset_index(drop=True)
        tmp[hid_key] = id
        id += 1
        dff = dff.append(tmp,ignore_index=True)
    timestamp= pd.to_datetime(dff[DATE_KEY] + " " + dff[TIME_KEY])

    dts = np.zeros(len(dff))
    for i in range(len(dff)):
        dt = timestamp[i] - timestamp[0]
        dts[i] = dt.seconds
    dff[dt_key] = dts
    dff=dff.drop(columns=[DATE_KEY,TIME_KEY,"<VOL>"])

    exf = extract_features(dff, column_id=hid_key, column_sort=dt_key)
    drop_col=[]
    for key in exf.keys():
        if exf[key].isnull().any() or np.isinf(exf[key]).any():
            drop_col.append(key)

    exf = exf.drop(columns=drop_col)

    return exf

def minmax(df,time_id,lookback):
    Log(LOG_INFO) << "Computing minmax with lookback: %d" % lookback
    minret = []
    maxret = []
    min_dist=[]
    max_dist=[]
    # pdb.set_trace()
    for tid in time_id:
        price = df[OPEN_KEY][tid]
        past_hi = df[HIGH_KEY][tid-lookback:tid].values
        past_lw = df[LOW_KEY][tid-lookback:tid].values
        hi = np.max(past_hi)/price - 1.
        lw = np.min(past_lw)/price - 1.
        minret.append(lw)
        maxret.append(hi)
        min_dist.append(lookback - np.argmin(past_lw))
        max_dist.append(lookback - np.argmax(past_hi))


    minret = np.array(minret)
    maxret = np.array(maxret)
    min_dist = np.array(min_dist)
    max_dist = np.array(max_dist)
    return np.hstack((minret.reshape(-1,1),maxret.reshape(-1,1),min_dist.reshape(-1,1),max_dist.reshape(-1,1)))

def basic_features_training(prices,tv,tm,lookback): # features based on hours, instead of min bars
    rx = np.diff(np.log(prices))

    Log(LOG_INFO) << "Computing basic features ..."
    nsample = len(prices) - lookback
    nf = 5
    fm = np.zeros((nsample, nf))
    label = np.ones(nsample) * np.nan
    for i in range(nsample):
        past = rx[i:i + lookback]
        # fm[i,:-3] = past
        # fm[i,-3] = np.mean(past)
        # fm[i,-2] = np.std(past)
        # fm[i,-1] = fm[i,-3]/fm[i,-2]


        tvp =  tv[i:i + lookback]
        fm[i, 0] = np.mean(past)
        fm[i, 1] = np.std(past)
        fm[i, 2] = np.mean(tvp)
        fm[i, 3] = np.std(tvp)
        fm[i, 4] = tm[i + lookback - 1].hour

    # print("feature dim: ", fm.shape)
    # print("actual labels: ", len(label))

    return fm

def prepare_features(fexconf, df, time_id):
    df[MID_KEY] = (df[HIGH_KEY] + df[LOW_KEY])*0.5
    prices = df[OPEN_KEY].values[time_id]
    tv = df[TICKVOL_KEY].values[time_id]
    tm = pd.to_datetime(df[TIME_KEY].values[time_id])
    lookback = fexconf.getLookback(BASICS_KEY)
    fm= basic_features_training(prices,tv,tm,lookback)
    used_time_id = time_id[lookback:]

    ################## additional features ###############
    if fexconf.isFeatureEnabled(VOLATILITY_KEY):
        fa = volatility(df, used_time_id, fexconf.getLookback(VOLATILITY_KEY),fexconf.getLongLookback(VOLATILITY_KEY))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled("DEMA"):
        fa = dema(df, used_time_id, fexconf.getLookback("DEMA"), fexconf.getLongLookback("DEMA"))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled(KAMA_KEY):
        fa = kama(df, used_time_id, fexconf.getLookback(KAMA_KEY), fexconf.getLongLookback(KAMA_KEY))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled(WMA_KEY):
        fa = wma(df, used_time_id, fexconf.getLookback(WMA_KEY), fexconf.getLongLookback(WMA_KEY))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled(BBANDS_KEY):
        fa = bbands(df, used_time_id, fexconf.getLookback(BBANDS_KEY))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled(MACD_KEY):
        fa = macd(df, used_time_id, fexconf.getLookback(MACD_KEY), fexconf.getLongLookback(MACD_KEY),fexconf.getFeatureParam(MACD_KEY,"DIF"))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled("MOMENTUM"):
        fa = momentum(df,used_time_id,fexconf.getLookback("MOMENTUM"))
        fm = np.hstack((fm,fa))
        fa = momentum(df, used_time_id, fexconf.getLongLookback("MOMENTUM"))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled("RSI"):
        fa = rsi(df,used_time_id,fexconf.getLookback("RSI"),fexconf.getLongLookback("RSI"))
        fm = np.hstack((fm,fa))

    if fexconf.isFeatureEnabled("CCI"):
        fa = cci(df,used_time_id,fexconf.getLookback("CCI"), fexconf.getLongLookback("CCI"))
        fm = np.hstack((fm,fa))

    if fexconf.isFeatureEnabled("RSV"):
        fa = rsv(df,used_time_id,fexconf.getLookback("RSV"))
        fm = np.hstack((fm,fa))

    if fexconf.isFeatureEnabled("SLOPE"):
        fa = slope(df,used_time_id,fexconf.getLookback("SLOPE"),fexconf.getLongLookback("SLOPE"))
        fm = np.hstack((fm,fa))

    if fexconf.isFeatureEnabled("TSFRESH"):
        fa = tsfresh(df, used_time_id, fexconf.getLookback("TSFRESH"))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled("ARIMA"):
        p = fexconf.getFeatureParam("ARIMA","P")
        d = fexconf.getFeatureParam("ARIMA","D")
        q = fexconf.getFeatureParam("ARIMA","Q")
        fa = arima(df,used_time_id,fexconf.getLookback("ARIMA"),p,d,q)
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled("MINMAX"):
        fa = minmax(df,used_time_id,fexconf.getLookback("MINMAX"))
        fm = np.hstack((fm,fa))
        fa = minmax(df, used_time_id, fexconf.getLongLookback("MINMAX"))
        fm = np.hstack((fm, fa))

        
    return fm, used_time_id, lookback

def split_dataset(fm, label, test_size):
    if test_size == 0:
        x_train = fm
        y_train = label
        x_test = np.array([])
        y_test = x_test
    else:
        x_train = fm[:-test_size, :]
        y_train = label[:-test_size]
        x_test = fm[-test_size:, :]
        y_test = label[-test_size:]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    Log(LOG_INFO) << "train and test sets splitted"

    return x_train, y_train, x_test, y_test, scaler

def split_dataset_by_dates(df, fm, labels,time_id,start_time,end_time):
    idx = len(time_id)-1
    id_s = -1
    id_e = -1 # not included
    while idx>=0:
        tid = time_id[idx]
        t0 = df[TIMESTAMP_KEY][tid]
        dt = t0 - end_time
        #print(t0, end_time, dt.total_seconds())
        if dt.total_seconds() <= 0:
            id_e = idx+1
            break
        idx-=1
    while 1:
        tid = time_id[idx]
        t0 = df[TIMESTAMP_KEY][tid]
        dt = t0 - start_time
        if dt.total_seconds() <= 0:
            id_s = idx+1
            break
        idx -= 1

    test_size = id_e - id_s
    x_train = fm[:id_s,:]
    y_train = labels[:id_s]
    x_test  = fm[id_s:id_e]
    y_test  = labels[id_s:id_e]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    tid_s = time_id[id_s]
    tid_e = time_id[id_e-1]
    Log(LOG_INFO) << "train and test sets splitted"
    Log(LOG_INFO) << "Test size: %d" % test_size
    Log(LOG_INFO) << "start date of test: " + df[DATE_KEY][tid_s] + " " + df[TIME_KEY][tid_s]
    Log(LOG_INFO) << "end   date of test: " + df[DATE_KEY][tid_e] + " " + df[TIME_KEY][tid_e]

    return x_train,y_train,x_test,y_test, id_s, scaler