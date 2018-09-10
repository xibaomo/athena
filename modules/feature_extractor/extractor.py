'''
Created on Sep 3, 2018

@author: fxua
'''

import gzip
import cPickle

from modules.basics.common.logger import *
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        self.trainFeatureMatrix = None
        self.trainTargets = None
        self.testFeatureMatrix = None
        self.testTargets = None
        return
    
    def save(self,filename,model):
        stream=gzip.open(filename,"wb")
        cPickle.dump(model,stream)
        stream.close()
        return
    
    def load(self,filename):
        stream = gzip.open(filename)
        model = cPickle.load(stream)
        stream.close()
        return model
    
    def getTrainFeatureMatrix(self):
        if self.trainFeatureMatrix is None:
            self.extractTrainFeatures()
            
        return self.trainFeatureMatrix
    
    def setTrainFeatureMatrix(self,fm):
        self.trainFeatureMatrix = fm
        return
    
    def getTrainTargets(self):
        return self.trainTargets
    
    def setTrainTargets(self,targets):
        self.trainTargets = targets
        return
    
    def getTestFeatureMatrix(self):
        if self.testFeatureMatrix is None:
            self.extractTestFeatures()
            
        return self.testFeatureMatrix 
    
    def setTestFeatureMatrix(self,fm):
        self.testFeatureMatrix = fm
        return
    
    def getTestTargets(self):
        return self.testTargets
    
    def setTestTargets(self,targets):
        self.testTargets = targets
        return
        
    
    def prepare(self,args=None):
        return
    
    def extractTrainFeatures(self):
        Log(LOG_FATAL) << "Should be implemented in concrete extractors"
        return
    
    def extractTestFeatures(self):
        Log(LOG_FATAL) << "Should be implemented in concrete extractors"
        return
    
