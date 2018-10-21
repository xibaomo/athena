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
    
    def computeRSI(self):
        rsi = talib.RSI(self.prices,timeperiod=self.config.getRSIPeriod())
        
        nullID = np.where(np.isnan(rsi))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        self.rawFeatures['RSI'] = rsi
        return
        
    def computeFeatures(self,featureNames):
        FeatureCalculatorSwitcher = {
        "DMA": self.computeDMA,
        "RSI": self.computeRSI
        }
        for f in featureNames:
            FeatureCalculatorSwitcher[f]()
    
    def getTotalFeatureMatrix(self):
        data = self.rawFeatures.values[len(self.nullID)+1:,:]
        labels = self.labels[len(self.nullID)+1:]
        
        if data.shape[0] != len(labels):
            Log(LOG_FATAL) << "Samples inconsistent with labels"
        return data,labels
        
        
        
        
        