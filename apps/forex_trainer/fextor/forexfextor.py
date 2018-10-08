'''
Created on Oct 1, 2018

@author: fxua
'''
from modules.forex_utils.common import *
from modules.basics.common.logger import *
from modules.feature_extractor.extractor import FeatureExtractor
from apps.forex_trainer.fextor.forexfexconf import ForexFexConfig
from modules.forex_utils.feature_calculator import FeatureCalculator
import pandas as pd
import numpy as np
class ForexFextor(FeatureExtractor):
    '''
    classdocs
    '''


    def __init__(self,foxconfig):
        '''
        Constructor
        '''
        self.config = ForexFexConfig(foxconfig)
        self.featureCalculator = FeatureCalculator(self.config)
        self.allTicks = None
        self.prices = None
        self.labels = None
        self.testSize = self.config.getTestPeriod()*ONEDAY/self.config.getSampleRate()
        Log(LOG_INFO) << "Test size: %d" % (self.testSize)
        return
    
    def loadTickFile(self):
        self.allTicks = pd.read_csv(self.config.getTickFile())
        self.prices = self.allTicks['price']
        self.labels = self.allTicks['label']
        
        self.featureCalculator.loadPriceLabel(self.prices,self.labels)
        return
    
    def computeFeatures(self):
        self.featureCalculator.computeFeatures(self.config.getFeatureList())
        return
        
    
    
            