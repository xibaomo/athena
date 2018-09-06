'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *

class MLEngineCore(object):
    
    def __init__(self,est=None):
        self.estimator = est 
        return
    
    def loadEstimator(self,est):
        self.estimator = est
        return
    
    def getEstimator(self):
        return self.estimator
    
    def train(self,feature_matrix,targets):
        Log(LOG_FATAL) << "Should be implemented in concrete class"
        return
    
    def predict(self,feature_matrix):
        Log(LOG_FATAL) << "Should be implemented in concrete class"
        return
    
    def getPredictedValues(self):
        Log(LOG_FATAL) << "Should be implemented in concrete class"
        return
    
    def getPredictedLabels(self):
        Log(LOG_FATAL) << "Should be implemented in concrete class"
        return
    