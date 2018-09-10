'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *

class MLEngineCore(object):
    
    def __init__(self, est=None):
        self.estimator = est 
        return
    
    def loadEstimator(self,est):
        self.estimator = est
        return
    
    def getEstimator(self):
        return self.estimator
    
    def train(self,fm,targets):        
        self.estimator.fit(fm,targets)
        return
    
    def predict(self,fm):       
        self.predictedTargets = self.estimator.predict(fm)
        return
    
    def getPredictedTargets(self):
        return self.predictedTargets
    