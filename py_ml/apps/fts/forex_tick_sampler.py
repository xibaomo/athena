'''
Created on Oct 2, 2018

@author: fxua
'''
import numpy as np
from apps.app import App
from apps.fts.ftsconf import FTSConfig
import csv
from dateutil import parser 
import pandas as pd
from modules.basics.common.logger import *
import pandas as pd
from modules.forex_utils.common import *

dateHeader = '<DATE>'
timeHeader = '<TIME>'
askHeader = '<ASK>'
bidHeader = '<BID>'
idHeader = 'OID'
fulltimeHeader = dateHeader+"_" + timeHeader

class ForexTickSampler(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexTickSampler,self).__init__()
        self.config = FTSConfig()
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
            item['time'] = sample[fulltimeHeader].to_pydatetime()
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
                sample[idHeader] = curInd
                ticks.append(sample)

            dt = t - prev
            if dt.total_seconds() < self.config.getSampleRate() or np.isnan(sample[opt]):
                continue
            sample[idHeader] = curInd
            ticks.append(sample)
            prev = t
            
        Log(LOG_INFO) << "Sampled ticks: %d" % len(ticks)
        return ticks
    
    def makeBuyLabels(self):
        Log(LOG_INFO) << "Labeling buy ticks ..."
        tp = self.config.getTakeProfitPoint()*self.config.getPointValue()
        sl = self.config.getStopLossPoint()*self.config.getPointValue()
        buyLabels = []
        eid = len(self.allTicks)

        asks = []
        time_buy = []
        k=0
        for bt in self.buyTicks:
            pos = bt['ask']
            label = None
            k+=1
            sid = bt[idHeader]+1
            Log(LOG_DEBUG) <<"Buy pos: %.5f" % pos
            
            for nk in range(sid,eid):
                tk = self.allTicks[nk]
                dt = tk['time'] - bt['time']
                if dt.total_seconds() <= 0:
                    continue
                if np.isnan(tk['bid']):
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
                time_buy.append((bt['time']))
                asks.append(pos)
                buyLabels.append(label)
                if k%500 == 0:
                    Log(LOG_INFO) <<"%d ticks labeled" % k
                
                Log(LOG_DEBUG) <<"Tick %d labeled %d" % (k,label)
        
        self.df_buy['time'] = time_buy
        self.df_buy['price'] = asks
        self.df_buy['label'] = buyLabels
        
        Log(LOG_INFO) << "Buy ticks are labeled."
        return 
        
    def makeSellLabels(self):
        Log(LOG_INFO) << "Labeling sell ticks ..."
        tp = self.config.getTakeProfitPoint()*self.config.getPointValue()
        sl = self.config.getStopLossPoint()*self.config.getPointValue()
        sellLabels = []
        eid = len(self.allTicks)

        time_sell = []
        bids = []
        k=0
        for bt in self.sellTicks:
            pos = bt['bid']
            label = None
            sid = bt[idHeader]+1
            k+=1
            
            for nk in range(sid,eid):
                tk = self.allTicks[nk]
                dt = tk['time'] - bt['time']
                if dt.total_seconds() <= 0:
                    continue
                if np.isnan(tk['ask']):
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
                if k%500 == 0:
                    Log(LOG_INFO) <<"%d ticks labeled" % k
        
        self.df_sell['time'] = time_sell
        self.df_sell['price'] = bids
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
        buyfile = self.config.getFXSymbol() + "_buy.csv"
        self.df_buy.to_csv(buyfile,index=False)
        sellfile = self.config.getFXSymbol() + "_sell.csv"
        self.df_sell.to_csv(sellfile,index=False)
        return
    
    