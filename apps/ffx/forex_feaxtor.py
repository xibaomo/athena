'''
Created on Oct 2, 2018

@author: fxua
'''
import numpy as np
import pandas as pd
from apps.app import App
from apps.ffx.ffxconf import FFXConfig
import copy
from dateutil import parser 
import pandas as pd
from modules.basics.common.logger import *


ONEMIN = 60
HALFMIN = 30
ONEHOUR = 60*ONEMIN
ONEDAY = 24*ONEHOUR
ONEWEEK = 7*ONEDAY
EXPIRE_PERIOD = ONEDAY
isLoss = 1
isProfit = 0
askHeader = '<ASK>'
bidHeader = '<BID>'
dateHeader = '<DATE>'
timeHeader = '<TIME>'
fulltimeHeader = dateHeader+'_'+timeHeader
labelHeader = 'LABEL'

class ForexFex(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexFex,self).__init__()
        self.config = FFXConfig()
        self.allTicks = None
        self.buyTicks = None
        self.sellTicks = None
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
        
    def loadTickFile(self):
        Log(LOG_INFO) << "Loading tick file ..."
        self.allTicks = pd.read_csv(self.config.getTickFile(),sep='\t',
                                    parse_dates=[[dateHeader,timeHeader]],
                                    nrows=1000)
                
        self.allTicks = self.allTicks.drop(columns=['<LAST>','<VOLUME>'])
        
        Log(LOG_INFO) << "Tick file loaded: %d" % self.allTicks.shape[0]
        return
    
    def _extractValidTicks(self,opt=askHeader):
        ticks=[]
        prev = None

        for m in range(self.allTicks.shape[0]):
            sample = self.allTicks.loc[m,:]
            t = sample[fulltimeHeader]
            if prev is None and not np.isnan(sample[opt]):
                prev = t
                sm = copy.deepcopy(sample)
                sm['oid'] = m
                sm[labelHeader] = -1
                ticks.append(sm)

            dt = t - prev
            if dt.total_seconds() < HALFMIN or np.isnan(sample[opt]):
                continue
            sm = copy.deepcopy(sample)
            sm['oid'] = m
            sm[labelHeader] = -1
            ticks.append(sm)
            prev = t
            
        df = pd.DataFrame(ticks)
        return df
    
    def cleanNullLabels(self,df):
        nl = []
        for m in range(df.shape[0]):
            f = df.loc[m,:]
            if f[labelHeader] == -1:
                nl.append(m)
                
        df=df.drop(nl)
        df = df.reset_index(drop=True)
        return df
    
    def makeBuyLabels(self):
        Log(LOG_INFO) << "Labeling buy ticks ..."
        tp = self.config.getTakeProfitPoint()*self.config.getPointValue()
        sl = self.config.getStopLossPoint()*self.config.getPointValue()
        eid = self.allTicks.shape[0]
        
        for m in range(self.buyTicks.shape[0]):
            tick = self.buyTicks.loc[m,:]
            pos = tick[askHeader]
            sid = tick['oid']+1
            
            Log(LOG_INFO) << "buy pos: %.5f" % pos
            for n in range(sid,eid):
                tk = self.allTicks.loc[n,:]
                dt = tk[fulltimeHeader] - tick[fulltimeHeader]
                bid = tk[bidHeader]
                if np.isnan(bid):
                    continue
                if pos - bid >= sl:
                    self.buyTicks.loc[m,labelHeader] = isLoss
                    Log(LOG_INFO) <<"loss: %.5f after %d sec" % (bid-pos,dt.total_seconds())
                    break
                if bid - pos >= tp:
                    self.buyTicks.loc[m,labelHeader] = isProfit
                    Log(LOG_INFO) <<"profit: %.5f after %d sec" % (bid-pos,dt.total_seconds())
                    break;
                if dt.total_seconds() >= EXPIRE_PERIOD:
                    self.buyTicks.loc[m,labelHeader] = isLoss
                    Log(LOG_INFO) <<"label expire after %d sec" % (dt.total_seconds())
                    break
            Log(LOG_INFO) <<"Tick %d labeled as %d" % (m,self.buyTicks.loc[m,labelHeader])
        self.buyTicks = self.cleanNullLabels(self.buyTicks)
        Log(LOG_INFO) << "Buy ticks are labeled"
        return 
        
    def makeSellLabels(self):
        Log(LOG_INFO) << "Labeling sell ticks ..."
        tp = self.config.getTakeProfitPoint()*self.config.getPointValue()
        sl = self.config.getStopLossPoint()*self.config.getPointValue()
        eid = self.allTicks.shape[0]
        
        for m in range(self.sellTicks.shape[0]):
            tick = self.sellTicks.loc[m,:]
            pos = tick[bidHeader]
            sid = tick['oid']+1
            
            Log(LOG_INFO) << "sell pos: %.5f" % pos
            for n in range(sid,eid):
                tk = self.allTicks.loc[n,:]
                dt = tk[fulltimeHeader] - tick[fulltimeHeader]
                ask = tk[askHeader]
                if np.isnan(ask):
                    continue
                if ask - pos >= sl:
                    Log(LOG_INFO) <<"loss: %.5f after %d sec" % (pos-ask,dt.total_seconds())
                    self.sellTicks.loc[m,labelHeader] = isLoss
                    break
                if pos - ask >= tp:
                    Log(LOG_INFO) <<"profit: %.5f after %d sec" % (pos-ask,dt.total_seconds())
                    self.sellTicks.loc[m,labelHeader] = isProfit
                    break
                if dt.total_seconds() >= EXPIRE_PERIOD:
                    Log(LOG_INFO) <<"label expire after %d sec" % (dt.total_seconds())
                    self.sellTicks.loc[m,labelHeader] =  isLoss
                    break
            
            Log(LOG_INFO) <<"Tick %d labeled as %d" % (m,self.sellTicks.loc[m,labelHeader])    
        self.sellTicks = self.cleanNullLabels(self.sellTicks)
        
        Log(LOG_INFO) << "Sell ticks are labeled."
        return 
                    
    def prepare(self):
        self.loadTickFile()
        Log(LOG_INFO) << "Sampling buy ticks ..."
        self.buyTicks = self._extractValidTicks(askHeader)
        self.buyTicks = self.buyTicks.reset_index(drop=True)
        Log(LOG_INFO) << "Done. Buy ticks: %d" % self.buyTicks.shape[0]
        
        Log(LOG_INFO) << "Sampling sell ticks ..."
        self.sellTicks = self._extractValidTicks(bidHeader)
        self.sellTicks = self.sellTicks.reset_index(drop=True)
        Log(LOG_INFO) << "Done. Sell ticks: %d" % self.sellTicks.shape[0]
        
        self.makeBuyLabels()
        self.makeSellLabels()
        return
    
    def execute(self):
        return
    
    def finish(self):
        buyfile = self.config.getFeatureTag() + "_buy.csv"
        self.buyTicks.to_csv(buyfile,index=False,index_label=False)
        sellfile = self.config.getFeatureTag() + "_sell.csv"
        self.sellTicks.to_csv(sellfile,index=False,index_label=False)
        return
    
    