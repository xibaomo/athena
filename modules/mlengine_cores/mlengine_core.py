'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *

class MLEngineCore(object):
    
    def __init__(self, fextor, est=None):
        self.featureExtractor = fextor
        self.estimator = est 
        return
    
    def loadEstimator(self,est):
        self.estimator = est
        return
    
    def getEstimator(self):
        return self.estimator
    
    def getFeatureExtractor(self):
        return self.featureExtractor
    
    def train(self):
                
        fm = self.featureExtractor.getTrainFeatureMatrix()
        targets = self.featureExtractor.getTrainTargets()
        
        self.estimator.fit(fm,targets)
        return
    
    def predict(self):
        fm = self.featureExtractor.getTestFeatureMatrix()
        
        self.predictedTargets = self.estimator.predict(fm)
        return
    
    def getPredictedTargets(self):
        return self.predictedTargets
    