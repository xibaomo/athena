'''
Created on Oct 7, 2018

@author: fxua
'''
import talib
import numpy as np
import pandas as pd
from modules.basics.common.logger import *



class FeatureCalculator(object):
    '''
    classdocs
    '''


    def __init__(self,forexfexconfig):
        '''
        Constructor
        '''
        self.config = forexfexconfig
        self.rawFeatures = pd.DataFrame()
        self.nullID = np.array([])
        
        return
        
    def loadPriceLabel(self,price,label):
        self.prices = price 
        self.labels = label
        return
    
    def computeDMA(self):
        slowma = talib.MA(self.prices,timeperiod=self.config.getSlowPeriod())
        fastma = talib.MA(self.prices,timeperiod=self.config.getFastPeriod()) 
        dma = fastma - slowma
        nullID = np.where(np.isnan(dma))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        self.rawFeatures['DMA'] = dma
        return
    
    def computeMACD(self):
        macd,macdsignal,macdhist = talib.MACD(self.prices,
                                              fastperiod=self.config.getFastPeriod(),
                                              slowperiod=self.config.getSlowPeriod(),
                                              signalperiod=self.config.getSignalPeriod())
        nullID = np.where(np.isnan(macd))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
            
        self.rawFeatures['MACD'] = macd
        self.rawFeatures['MACDSIGNAL'] = macdsignal
        return
    
    def computeRSI(self):
        srsi = talib.RSI(self.prices,timeperiod=self.config.getSlowPeriod())
        frsi = talib.RSI(self.prices,timeperiod=self.config.getFastPeriod())
        
        rsi = frsi-srsi
        nullID = np.where(np.isnan(rsi))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        self.rawFeatures['RSI'] = rsi
        return
        
    def computeCMO(self):
        scmo = talib.CMO(self.prices,timeperiod=self.config.getSlowPeriod())
        fcmo = talib.CMO(self.prices,timeperiod=self.config.getFastPeriod())
        cmo = fcmo - scmo
        self.removeNullID(cmo)
        self.rawFeatures['CMO']=cmo
        return
    
    def computeDROC(self):
        sroc = talib.ROC(self.prices,timeperiod=self.config.getSlowPeriod())
        froc = talib.ROC(self.prices,timeperiod=self.config.getFastPeriod())
        droc = froc - sroc 
        self.removeNullID(droc)
        self.rawFeatures['ROC']=droc
        return
    
    def computeDEMA(self):
        sema = talib.EMA(self.prices,timeperiod=self.config.getSlowPeriod())
        fema = talib.EMA(self.prices,timeperiod=self.config.getFastPeriod())
        dema = fema - sema 
        self.removeNullID(dema)
        self.rawFeatures['EMA'] = dema
        return
    
    def computeDKAMA(self):
        skama = talib.KAMA(self.prices,timeperiod=self.config.getSlowPeriod())
        fkama = talib.KAMA(self.prices,timeperiod=self.config.getFastPeriod())
        dkama = fkama - skama 
        self.removeNullID(dkama)
        self.rawFeatures['KAMA'] = dkama
        return
        
    def computeFeatures(self,featureNames):
        FeatureCalculatorSwitcher = {
        "DMA": self.computeDMA,
        "RSI": self.computeRSI,
        "MACD": self.computeMACD,
        "CMO": self.computeCMO,
        "ROC": self.computeDROC,
        "EMA": self.computeDEMA,
        "KAMA": self.computeDKAMA
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
        
        df.to_csv("features.csv")
        if data.shape[0] != len(labels):
            Log(LOG_FATAL) << "Samples inconsistent with labels"
        return data,labels
        
        
        
        
        