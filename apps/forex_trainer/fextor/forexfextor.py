'''
Created on Oct 1, 2018

@author: fxua
'''
from modules.feature_extractor.extractor import FeatureExtractor
from apps.forex_trainer.fextor.forexfexconf import ForexFexConfig
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
        self.allTicks = None
        self.prices = None
        self.labels = None
        return
    
    def loadTickFile(self):
        self.allTicks = pd.read_csv(self.config.getTickFile())
        self.prices = self.allTicks['price']
        self.labels = self.allTicks['label']
        
    
    
            