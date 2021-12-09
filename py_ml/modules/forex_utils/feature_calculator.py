'''
Created on Oct 7, 2018

@author: fxua
'''
import pdb
import talib
import numpy as np
import pandas as pd
import copy
from modules.basics.common.logger import *

class FeatureCalculator(object):
    '''
    classdocs
    '''


    def __init__(self,forexfexconfig=None):
        '''
        Constructor
        '''
        self.config = forexfexconfig
        self.rawFeatures = pd.DataFrame()
        self.nullID = np.array([])
        
        self.prices=np.array([])
        self.fastPeriod = None
        self.slowPeriod = None
        if self.config is not None:
            self.fastPeriod = self.config.getFastPeriod()
            self.slowPeriod = self.config.getSlowPeriod()
        
        return
    
    def __str__(self):
        return "Feature calculator:  " + str(self.fastPeriod) + " " + str(self.slowPeriod)
    
    def resetFeatureTable(self):
        self.rawFeatures = pd.DataFrame()
        return
    
    def setPeriods(self,fast,slow):
        self.slowPeriod = slow
        self.fastPeriod = fast
        if slow < fast:
            Log(LOG_FATAL) << "fast period is longer than slow period: %d vs %d" % (fast,slow)
        return
        
    def loadPrice(self,price):
        self.prices = price 
        print "Total ticks: " + str(len(self.prices))
        return
    
    def appendPrice(self,p):
        self.prices = np.append(self.prices, p)
        print "Total ticks: " + str(len(self.prices))
        return
    
    def getLatestfeatures(self):
        f = self.rawFeatures.iloc[-1,:].values
        return f
    
    def loadPriceLabel(self,price,label):
        self.prices = price 
        self.labels = label
        return
    
    def computeDMA(self):
        slowma = talib.MA(self.prices,timeperiod=self.slowPeriod)
        fastma = talib.MA(self.prices,timeperiod=self.fastPeriod) 
        dma = fastma - slowma
        dma = slowma
        nullID = np.where(np.isnan(dma))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        self.rawFeatures['DMA'] = dma
        return
    
    def computeMACD(self):
        macd,macdsignal,macdhist = talib.MACD(self.prices,
                                              fastperiod=self.fastPeriod,
                                              slowperiod=self.slowPeriod,
                                              signalperiod=self.config.getSignalPeriod())
        nullID = np.where(np.isnan(macd))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
            
        self.rawFeatures['MACD'] = macd
        self.rawFeatures['MACDSIGNAL'] = macdsignal
        return
    
    def computeRSI(self):
        srsi = talib.RSI(self.prices,timeperiod=self.slowPeriod)
        frsi = talib.RSI(self.prices,timeperiod=self.fastPeriod)
        
        rsi = frsi-srsi
        rsi = srsi
        nullID = np.where(np.isnan(rsi))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        self.rawFeatures['RSI'] = rsi
        return
        
    def computeCMO(self):
        scmo = talib.CMO(self.prices,timeperiod=self.slowPeriod)
        fcmo = talib.CMO(self.prices,timeperiod=self.fastPeriod)
        cmo = fcmo - scmo
        cmo = scmo
        self.removeNullID(cmo)
        self.rawFeatures['CMO']=cmo
        return
    
    def computeDROC(self):
        sroc = talib.ROC(self.prices,timeperiod=self.slowPeriod)
        froc = talib.ROC(self.prices,timeperiod=self.fastPeriod)
        droc = froc - sroc 
        
        droc = sroc
        self.removeNullID(droc)
        self.rawFeatures['ROC']=droc
        return
    
    def computeDEMA(self):
        sema = talib.EMA(self.prices,timeperiod=self.slowPeriod)
        fema = talib.EMA(self.prices,timeperiod=self.fastPeriod)
        dema = fema - sema 
        self.removeNullID(dema)
        self.rawFeatures['EMA'] = dema
        return
    
    def computeDKAMA(self):
        skama = talib.KAMA(self.prices,timeperiod=self.slowPeriod)
        fkama = talib.KAMA(self.prices,timeperiod=self.fastPeriod)
        dkama = fkama - skama 
        self.removeNullID(dkama)
        self.rawFeatures['KAMA'] = dkama
        return
    
    def computeDLag(self):
        flag = talib.LINEARREG(self.prices,timeperiod=self.fastPeriod)
        slag = talib.LINEARREG(self.prices,timeperiod=self.slowPeriod)
        dlag = flag - slag
        self.removeNullID(dlag)
        self.rawFeatures['LAG'] = dlag
        
    def computeDStdv(self):
        fstd = talib.STDDEV(self.prices,self.fastPeriod)
        sstd = talib.STDDEV(self.prices,self.slowPeriod)
        stdv = fstd-sstd
        
        stdv = fstd
        self.removeNullID(stdv)
        self.rawFeatures['STDV'] = stdv
        
    def computeFeatures(self,featureNames):
        FeatureCalculatorSwitcher = {
        "DMA": self.computeDMA,
        "RSI": self.computeRSI,
        "MACD": self.computeMACD,
        "CMO": self.computeCMO,
        "ROC": self.computeDROC,
        "EMA": self.computeDEMA,
        "KAMA": self.computeDKAMA,
        "LAG": self.computeDLag,
        "STDV": self.computeDStdv
        }
        for f in featureNames:
            FeatureCalculatorSwitcher[f]()
    
    def removeNullID(self,ind):
        nullID = np.where(np.isnan(ind))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        return
    def getTotalFeatureMatrix(self):
        data = self.rawFeatures.values[len(self.nullID)+1:,:]
        labels = self.labels[len(self.nullID)+1:]
        
        df = pd.DataFrame(data,columns=self.rawFeatures.keys())
        df['label']=labels
        
        df.to_csv("features.csv",index=False)
        if data.shape[0] != len(labels):
            Log(LOG_FATAL) << "Samples inconsistent with labels"
        return data,labels
        
        
        
        
        