'''
Created on Oct 2, 2018

@author: fxua
'''
import numpy as np
from apps.app import App
from apps.ffx.ffxconf import FFXConfig
import csv
from dateutil import parser 
import pandas as pd
from modules.basics.common.logger import *
import pandas as pd

dateHeader = '<DATE>'
timeHeader = '<TIME>'
askHeader = '<ASK>'
bidHeader = '<BID>'
fulltimeHeader = dateHeader+"_" + timeHeader
ONEMIN = 60
HALFMIN = 30
ONEHOUR = 60*ONEMIN
ONEDAY = 24*ONEHOUR
ONEWEEK = 7*ONEDAY
isLoss = 1
isProfit = 0
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
        self.allTicks = []
        self.buyTicks = []
        self.buyLabels = []
        self.sellTicks = []
        self.sellLabels= []
        self.df_sell = pd.DataFrame()
        self.df_buy = pd.DataFrame()
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def loadTickFile(self):
        Log(LOG_INFO) << "Loading tick file ..."
        df = pd.read_csv(self.config.getTickFile(),sep='\t',
                         parse_dates=[[dateHeader,timeHeader]])
        
        for _,sample in df.iterrows():
            item = {}
            item['time'] = sample[fulltimeHeader]
            item['ask'] = sample[askHeader]
            item['bid'] = sample[bidHeader]
            self.allTicks.append(item)
            
        Log(LOG_INFO) << "All ticks loaded: %d" % len(self.allTicks)
        return
        
    def _extractValidTicks(self,opt='ask'):
        ticks=[]
        prev = None
        curInd = -1
        for sample in self.allTicks:
            curInd+=1
            t = sample['time']
            if prev is None and not np.isnan(sample[opt]):
                prev = t
                sample['index'] = curInd
                ticks.append(sample)

            dt = t - prev
            if dt.total_seconds() < self.config.getSampleRate() or np.isnan(sample[opt]):
                continue
            sample['index'] = curInd
            ticks.append(sample)
            prev = t
            
        Log(LOG_INFO) << "Sampled ticks: %d" % len(ticks)
        return ticks
    
    def makeBuyLabels(self):
        Log(LOG_INFO) << "Labeling buy ticks ..."
        tp = self.config.getTakeProfitPoint()*self.config.getPointValue()
        sl = self.config.getStopLossPoint()*self.config.getPointValue()
        buyLabels = []

        asks = []
        time_buy = []
        k=0
        for bt in self.buyTicks:
            pos = bt['ask']
            label = None
            
            Log(LOG_DEBUG) <<"Buy pos: %.5f" % pos
            for tk in self.allTicks:
                dt = tk['time'] - bt['time']
                if dt.total_seconds() <= 0:
                    continue
                if tk['bid'] is None:
                    continue
                if tk['bid'] - pos >= tp:
                    label = isProfit
                    Log(LOG_DEBUG) << "Profit: %.5f after %d sec" % (tk['bid'],dt.total_seconds())
                    break
                if pos - tk['bid'] >= sl:
                    label = isLoss
                    Log(LOG_DEBUG) << "Loss: %.5f after %d sec" % (tk['bid'],dt.total_seconds())
                    break
                if dt.total_seconds() > self.config.getExpiryPeriod()*ONEDAY:
                    label = isLoss 
                    break
            
            if label is not None:
                time_buy.append(str(bt['time']))
                asks.append(pos)
                buyLabels.append(label)
                k+=1
                Log(LOG_INFO) <<"Tick %d labeled %d" % (k,label)
        
        self.df_buy['time'] = time_buy
        self.df_buy['ask'] = asks
        self.df_buy['label'] = buyLabels
        
        Log(LOG_INFO) << "Buy ticks are labeled."
        return 
        
    def makeSellLabels(self):
        Log(LOG_INFO) << "Labeling sell ticks ..."
        tp = self.config.getTakeProfitPoint()*self.config.getPointValue()
        sl = self.config.getStopLossPoint()*self.config.getPointValue()
        sellLabels = []

        time_sell = []
        bids = []
        for bt in self.sellTicks:
            pos = bt['bid']
            label = None
            for tk in self.allTicks:
                dt = tk['time'] - bt['time']
                if dt.total_seconds() <= 0:
                    continue
                if tk['ask'] is None:
                    continue
                if pos - tk['ask'] >= tp:
                    label = isProfit
                    break
                if tk['ask'] - pos >= sl:
                    label = isLoss
                    break
                if dt.total_seconds() > self.config.getExpiryPeriod()*ONEDAY:
                    label = isLoss 
                    break
                
            if label is not None:
                time_sell.append(str(bt['time']))
                bids.append(pos)
                sellLabels.append(label)
        
        self.df_sell['time'] = time_sell
        self.df_sell['bid'] = bids
        self.df_sell['label'] = sellLabels
        
        Log(LOG_INFO) << "Sell ticks are labeled."
        return 
                    
    def prepare(self):
        self.loadTickFile()
        Log(LOG_INFO) << "Sampling buy ticks ..."
        self.buyTicks = self._extractValidTicks('ask')
        Log(LOG_INFO) << "Done"
        
        Log(LOG_INFO) << "Sampling sell ticks ..."
        self.sellTicks = self._extractValidTicks('bid')
        Log(LOG_INFO) << "Done"
        self.makeBuyLabels()
        self.makeSellLabels()
        return
    
    def execute(self):
        return
    
    def finish(self):
        buyfile = self.config.getFeatureTag() + "_buy.csv"
        self.df_buy.to_csv(buyfile)
        sellfile = self.config.getFeatureTag() + "_sell.csv"
        self.df_sell.to_csv(sellfile)
        return
    
    