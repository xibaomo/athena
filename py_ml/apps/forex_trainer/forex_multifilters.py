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
        num_fail = sum(self.fextor.getTestTargets())
        dream_profit = (testSize-num_fail) * self.config.getPointValue() * self.config.getTakeProfit()
        Log(LOG_INFO) << "%d transactions. Dream profit = $%.2f" % (testSize,dream_profit)
        
        self.workForce.train()
        
        fm = self.fextor.getTestFeatureMatrix();
        tar = self.fextor.getTestTargets()
        num_good,num_miss = self.workForce.filterBadPoints(fm,tar)
        
        self.profit = self.computeProfit(num_good, num_miss)
        
        Log(LOG_INFO) << "***************************";
        Log(LOG_INFO) << "********* Summary *********"
        Log(LOG_INFO) << "***************************";
        Log(LOG_INFO) << "Total transactions (original): %d" % (len(tar))
        badfrac = sum(tar)/float(len(tar)) 
        Log(LOG_INFO) << "good: %.1f%%, bad: %.1f%%" % ((1-badfrac)*100,badfrac*100)
        Log(LOG_INFO) << "Actual transactions: %d" % (num_good + num_miss)
        Log(LOG_INFO) << "Profit transactions: %d" % (num_good)
        Log(LOG_INFO) << "Loss transactions: %d" % (num_miss)
        Log(LOG_INFO) << "Total profit: %.2f" % self.profit
        Log(LOG_INFO) << "Dream profit: %.2f" % dream_profit
        Log(LOG_INFO) << "%.2f%% of dream profit taken" % (100*self.profit/dream_profit)
        
        return
    
    def computeProfit(self,num_good,num_miss):
        profitPerTran = self.config.getPointValue() * self.config.getTakeProfit()
        lossPerTran = self.config.getPointValue() * self.config.getStopLoss()
        
        profit = num_good * profitPerTran -  num_miss * lossPerTran
        
        return profit
    
    def finish(self):
        self.workForce.saveFilters(self.config.getModelPrefix(),self.profit)
        return
        