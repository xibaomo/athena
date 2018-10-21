'''
Created on Oct 1, 2018

@author: fxua
'''
from apps.app import App
from apps.forex_trainer.fxtconf import FxtConfig
from apps.forex_trainer.fextor.forexfextor import ForexFextor
from modules.appframeworks.cascadedfilters.overkillfilters.overkillfilters import OverkillFilters
from modules.basics.common.logger import *
class ForexMultiFilters(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexMultiFilters,self).__init__()
        self.config = FxtConfig()
        self.fextor = ForexFextor(self.config)
        self.workForce = OverkillFilters(self.fextor)
        
        Log(LOG_INFO) << "Forex multifilters created"
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):
        self.fextor.loadTickFile()
        self.fextor.computeFeatures()
        return
    
    def execute(self):
        
        testSize = len(self.fextor.getTestTargets())
        dream_profit = testSize * self.config.getPointValue() * self.config.getTakeProfit()
        Log(LOG_INFO) << "%d transactions. Dream profit = $%.2f" % (testSize,dream_profit)
        
        self.workForce.train()
        
        num_good,num_miss = self.workForce.filterBadPoints()
        
        profit = self.computeProfit(num_good, num_miss)
        
        Log(LOG_INFO) << "Profit transactions: %d" % num_good
        Log(LOG_INFO) << "Loss transactions: %d" % num_miss
        Log(LOG_INFO) << "Total profit: $%.2f" % profit
        Log(LOG_INFO) << "%.2f%% of dream profit taken" % (100*profit/dream_profit)
        
        return
    
    def finish(self):
        return
    
    def computeProfit(self,num_good,num_miss):
        profitPerTran = self.config.getPointValue() * self.config.getTakeProfit()
        lossPerTran = self.config.getPointValue() * self.config.getStopLoss()
        
        profit = num_good * profitPerTran -  num_miss * lossPerTran
        
        return profit
        