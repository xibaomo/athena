'''
Created on Dec 2, 2018

@author: fxua
'''
import talib 
import numpy as np
import pandas as pd
from modules.basics.common.logger import *


class BarFeatureCalculator(object):
    '''
    classdocs
    '''


    def __init__(self, config=None):
        '''
        Constructor
        '''
        self.config = config
        self.rawFeatures = pd.DataFrame()
        self.nullID = np.array([])
        
        self.lookback = self.config.getLookBack()
        self.allMinBars = pd.read_csv(self.config.getBarFile())
        self.open = self.allMinBars['OPEN']
        self.high = self.allMinBars['HIGH']
        self.low  = self.allMinBars['LOW']
        self.close = self.allMinBars['CLOSE']
        self.labels = self.allMinBars['LABEL']
        return
    
    def resetFeatureTable(self):
        self.rawFeatures = pd.DataFrame()
        return
    
    def computeFeatures(self,featureNames):
        BarFeatureSwitcher = {
            "MIDPRICE": self.compMidPrice,
            "KAMA" : self.compKAMA,
            "RSI" : self.compRSI,
            "WILLR" : self.compWILLR,
            "TRIX" : self.compTRIX,
            "ROC" : self.compROC,
            "AROONOSC" : self.compAROONOSC,
            "ADX" : self.compADX,
            "DX" : self.compDX,
            "CMO" : self.compCMO,
            "BETA" : self.compBETA,
            "BBANDS" : self.compBBANDS
        }
        
        for fn in featureNames:
            BarFeatureSwitcher[fn]()
    
    def removeNullID(self,ind):
        nullID = np.where(np.isnan(ind))[0]
        if len(nullID) > len(self.nullID):
            self.nullID = nullID
        return
    
    def compMidPrice(self):
        mp = talib.MIDPRICE(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(mp)
        
        self.rawFeatures['MIDPRICE'] = mp
        return
    
    def compKAMA(self):
        kama = talib.KAMA(self.close,timeperiod=self.lookback)
        self.removeNullID(kama)
        
        self.rawFeatures['KAMA'] = kama
        return
    
    def compRSI(self):
        rsi = talib.RSI(self.close,timeperiod=self.lookback)
        self.removeNullID(rsi)
        self.rawFeatures['RSI'] = rsi
        return
    
    def compWILLR(self):
        wr = talib.WILLR(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(wr)
        self.rawFeatures['WILLR'] = wr
        return
    
    def compTRIX(self):
        tx = talib.TRIX(self.close,timeperiod=self.lookback)
        self.removeNullID(tx)
        self.rawFeatures['TRIX'] = tx
        return
    
    def compROC(self):
        roc = talib.ROC(self.close,timeperiod=self.lookback)
        self.removeNullID(roc)
        self.rawFeatures['ROC'] = roc 
        return
    
    def compAROONOSC(self):
        ac = talib.AROONOSC(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(ac)
        self.rawFeatures['AROONOSC'] = ac
        return
    
    def compADX(self):
        adx = talib.ADX(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(adx)
        self.rawFeatures['ADX'] = adx 
        return
    
    def compATR(self):
        atr = talib.ATR(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(atr)
        self.rawFeatures['ATR'] = atr 
        return
    
    def compDX(self):
        dx = talib.DX(self.high,self.low,self.close,timeperiod=self.lookback)
        self.removeNullID(dx)
        self.rawFeatures['DX'] = dx
        return
    
    def compTSF(self):
        tsf = talib.TSF(self.close,timeperiod=self.lookback)
        self.removeNullID(tsf)
        self.rawFeatures['TSF'] = tsf
        return
    
    def compCMO(self):
        cmo = talib.CMO(self.close,timeperiod=self.lookback)
        self.removeNullID(cmo)
        self.rawFeatures['CMO'] = cmo 
        return
    
    def compBETA(self):
        beta = talib.BETA(self.high,self.low,timeperiod=self.lookback)
        self.removeNullID(beta)
        self.rawFeatures['BETA'] = beta
        return
    
    def compBBANDS(self):
        ub,mb,lb = talib.BBANDS(self.close,timeperiod=self.lookback)
        self.removeNullID(ub)
        self.rawFeatures['UPPERBAND'] = ub 
        self.rawFeatures['MIDDLEBAND'] = mb
        self.rawFeatures['LOWERBAND'] = lb
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
    
    
    
    