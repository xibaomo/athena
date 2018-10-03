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
        k = 0
        with open(self.config.getTickFile(),'r') as f:
            reader = csv.reader(f,delimiter='\t')
            for row in reader:
                if '<DATE>' in row:
                    continue
                sample={}
                timestr = row[0] + " " + row[1]
                t = parser.parse(timestr)
                                
                sample['time'] = t
                bid = row[2]
                if bid == "":
                    sample['bid'] = None
                else:
                    sample['bid'] = float(bid)
                    
                ask = row[3]
                if ask == "":
                    sample['ask'] = None
                else:
                    sample['ask'] = float(ask)
                    
                self.allTicks.append(sample)
                k+=1
                if k % 1e4 == 0:
                    Log(LOG_INFO) << "%d loaded" % k
                
        Log(LOG_INFO) << "Tick file loaded: %d" % len(self.allTicks)
        return
        
    def _extractValidTicks(self,opt='ask'):
        ticks=[]
        prev = None
        for sample in self.allTicks:
            t = sample['time']
            if prev is None and sample[opt] is not None:
                prev = t
                ticks.append(sample)

            dt = t - prev
            if dt.total_seconds() < HALFMIN or sample[opt] is None:
                continue
            ticks.append(sample)
            prev = t
        return ticks
    
    def makeBuyLabels(self):
        tp = self.config.getTakeProfitPoint()*self.config.getPointValue()
        sl = self.config.getStopLossPoint()*self.config.getPointValue()
        buyLabels = []
        buyticks = []
        asks = []
        time_buy = []
        for bt in self.buyTicks:
            pos = bt['ask']
            label = None
            for tk in self.allTicks:
                dt = tk['time'] - pos['time']
                if dt.total_seconds() <= 0:
                    continue
                if tk['bid'] is None:
                    continue
                if tk['bid'] - pos >= tp:
                    label = isProfit
                    break
                if pos - tk['bid'] >= sl:
                    label = isLoss
                    break
                if dt.total_seconds() > ONEWEEK:
                    label = isLoss 
                    break
            
            if label is not None:
                time_buy.append(str(bt['time']))
                asks.append(pos)
                buyLabels.append(label)
        
        self.df_buy['time'] = time_buy
        self.df_buy['ask'] = asks
        self.df_buy['label'] = buyLabels
        return 
        
    def makeSellLabels(self):
        tp = self.config.getTakeProfitPoint()*self.config.getPointValue()
        sl = self.config.getStopLossPoint()*self.config.getPointValue()
        sellLabels = []
        sellticks = []
        time_sell = []
        bids = []
        for bt in self.sellTicks:
            pos = bt['bid']
            label = None
            for tk in self.allTicks:
                dt = tk['time'] - pos['time']
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
                if dt.total_seconds() > ONEWEEK:
                    label = isLoss 
                    break
                
            if label is not None:
                time_sell.append(str(bt['time']))
                bids.append(pos)
                sellLabels.append(label)
        
        self.df_sell['time'] = time_sell
        self.df_sell['bid'] = bids
        self.df_sell['label'] = sellLabels
        return 
                    
    def prepare(self):
        self.loadTickFile()
        self.buyTicks = self._extractValidTicks('ask')
        self.sellTicks = self._extractValidTicks('bid')
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
    
    