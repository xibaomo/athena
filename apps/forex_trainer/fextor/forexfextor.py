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
#         self.testSize = self.config.getTestPeriod()*ONEDAY/self.config.getSampleRate()
        self.testSize = self.config.getTestSize()
        Log(LOG_INFO) << "Test size: %d" % (self.testSize)
        
        return
    
    def loadTickFile(self):
        self.allTicks = pd.read_csv(self.config.getTickFile())
        self.prices = self.allTicks['price'].values
        self.labels = self.allTicks['label'].values
        
        Log(LOG_INFO) << "Loaded ticks: %d" % self.allTicks.shape[0]
        if self.testSize > self.allTicks.shape[0]:
            Log(LOG_FATAL) << "Test size larger than all ticks."
            
        self.featureCalculator.loadPriceLabel(self.prices,self.labels)
        return
    
    def computeFeatures(self):
        self.featureCalculator.computeFeatures(self.config.getFeatureList())
        
        self.totalFeatureMatrix,self.labels = self.featureCalculator.getTotalFeatureMatrix()
        
        Log(LOG_INFO) << "Valid total ticks: %d" % len(self.labels)
        testSize = self.config.getTestSize()
        trainSize = len(self.labels) - testSize
        
        
        self.trainFeatureMatrix = self.totalFeatureMatrix[:trainSize,:]
        self.trainTargets = self.labels[:trainSize]
        self.testFeatureMatrix = self.totalFeatureMatrix[trainSize:,:]
        self.testTargets = self.labels[trainSize:]
        return
        
    
    
            