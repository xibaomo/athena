'''
Created on Sep 3, 2018

@author: fxua
'''

import gzip
import cPickle

from modules.basics.common.logger import *
import numpy as np
from sklearn.preprocessing.data import StandardScaler
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.basics.conf.generalconf import gGeneralConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

class FeatureExtractor(object):
    def __init__(self):
        self.trainFeatureMatrix = None
        self.trainTargets = None
        self.testFeatureMatrix = None
        self.testTargets = None
        self.verifyFeatureMatrix = None
        self.verifyTargets = None
        self.scaler = None

        if gGeneralConfig.isEnablePCA():
            self.pca = self.createPCA()

        if gMLEngineConfig.getEngineCoreType() >=2:
            Log(LOG_INFO) << "Neural network needs no scaler"
            return

        sf = gGeneralConfig.getLoadScalerFile()
        if sf == "":
            self.createScaler()
            Log(LOG_INFO) << "No scaler is given, created one"
        else:
            self.loadScaler(sf)
            Log(LOG_INFO) << "Scaler loaded: %s" % sf
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

    def loadScaler(self,filename):
        self.scaler = joblib.load(filename)
        return

    def createScaler(self):
        self.scaler = StandardScaler()
        return

    def saveScaler(self,filename):
        joblib.dump(self.scaler,filename)
        return
    
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

    def createPCA(self):
        pf = gGeneralConfig.getLoadPCAFile()
        if pf == "":
            pca = PCA(n_components=gGeneralConfig.getPCAComponents())
            Log(LOG_INFO) << "PCA is enabled with %d components" % gGeneralConfig.getPCAComponents()
        else:
            Log(LOG_INFO) << "loading pca not supported"

        return pca
