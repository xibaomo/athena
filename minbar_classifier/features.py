import pdb

import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from logger import *
from basics import *
import yaml

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
MA_KEY          = "MA_MISC"
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
    def isFeatureEnabled(self, key):
        if not key in self.fexDict:
            return False
        if self.fexDict[key][LOOKBACK_KEY] < 0:
            return False
        return True

def volatility(df, time_id, lookback = 2):
    ft = np.zeros((len(time_id), 2))
    for i in range(len(time_id)):
        tid = time_id[i]
        past_sd = df['STD'][tid-lookback:tid].values
        ft[i, 0] = np.mean(past_sd)
        ft[i, 1] = np.std(past_sd)
    return ft
def ma_misc(df, time_id, slk = 6, llk = 12):
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
    dif = prices[time_id] - kma_s[time_id]
    dkma = kma_s[time_id] - kma_l[time_id]
    return np.hstack((dif.reshape(-1, 1), dkma.reshape(-1, 1)))

def wma(df, time_id, slk = 3, llk = 24):
    Log(LOG_INFO) << "Computing wma with %d, %d" % (slk, llk)
    prices = df[OPEN_KEY].values
    kma_s = talib.WMA(prices, slk)
    kma_l = talib.WMA(prices, llk)
    dif = prices[time_id] - kma_s[time_id]
    dwma = kma_s[time_id] - kma_l[time_id]
    return np.hstack((dif.reshape(-1, 1), dwma.reshape(-1, 1)))

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

def basic_features_training(prices,tv,tm,lookback):
    #prices = df[OPEN_KEY].values[time_id]
    rx = np.diff(np.log(prices))
    #tv = df["<TICKVOL>"].values[time_id]
    #tm = pd.to_datetime(df[TIME_KEY].values[time_id])
    # tv = tv[:-1]
    # tm = tm[:-1]

    Log(LOG_INFO) << "Computing basic features ..."
    nsample = len(prices) - lookback
    nf = 5
    fm = np.zeros((nsample, nf))
    label = np.ones(nsample) * np.nan
    for i in range(nsample):
        past = rx[i:i + lookback]
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
    s = np.log(df[HIGH_KEY] / df[LOW_KEY])
    df[STD_KEY] = s  # volatility of each bar
    #fexconf = FexConfig(conf_file)

    prices = df[OPEN_KEY].values[time_id]
    tv = df[TICKVOL_KEY].values[time_id]
    tm = pd.to_datetime(df[TIME_KEY].values[time_id])
    lookback = fexconf.getLookback(BASICS_KEY)
    fm= basic_features_training(prices,tv,tm,lookback)
    used_time_id = time_id[lookback:]

    ################## additional features ###############
    if fexconf.isFeatureEnabled(VOLATILITY_KEY):
        fa = volatility(df, used_time_id, fexconf.getLookback(VOLATILITY_KEY))
        fm = np.hstack((fm, fa))

    if fexconf.isFeatureEnabled(MA_KEY):
        fa = ma_misc(df, used_time_id, fexconf.getLookback(MA_KEY), fexconf.getLongLookback(MA_KEY))
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
        fa = macd(df, used_time_id, fexconf.getLookback(MACD_KEY), fexconf.getLongLookback(MACD_KEY))
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

    return x_train, y_train, x_test, y_test, scaler
