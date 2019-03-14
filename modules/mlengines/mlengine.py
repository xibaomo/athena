'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *
from modules.basics.conf.generalconf import gGeneralConfig
from modules.basics.conf.mlengineconf import gMLEngineConfig
import numpy as np

class MLEngine(object):
    def __init__(self,engineCore=None):
        self.engineCore = engineCore
        self.predicted_labels = None
        return
    
    def loadEngineCore(self,engine_core):
        self.engineCore = engine_core
        return
    
    def train(self,fm,targets):
        Log(LOG_INFO) << "Start training model ..."
        self.engineCore.train(fm,targets)
        Log(LOG_INFO) << "Training is finished"
        
        acc,mr,crt = self.engineCore.getAccuracy(fm,targets)

        Log(LOG_INFO) << "Correct on train set: " + str(crt)
        Log(LOG_INFO) << "Accuracy on train set: " + str(acc)
        Log(LOG_INFO) << "Miss ratio on train set: " + str(mr)
        
        return
    
    def predict(self):
        Log(LOG_FATAL) << "This function should be implemented in concrete class"
        return

    def predict_array(self,arr,nrows,ncols):
        fm = np.array(arr)
        fm = fm.reshape(nrows,ncols)
        self.predict(fm)

        return

    def getPredictedTargets(self):
        pred = np.array(self.engineCore.getPredictedTargets())
        return pred

    def saveModel(self,mfn):
        self.engineCore.saveModel(mfn)
        return

    def loadModel(self,mfn):
        self.engineCore.loadModel(mfn)
        return

    def showEstimator(self):
        self.engineCore.showEstimator
        return

    def getEstimator(self):
        return self.engineCore.getEstimator()
    

        
