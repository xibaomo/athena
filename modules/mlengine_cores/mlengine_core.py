'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *
from modules.mlengine_cores.dnncomm.dnncreator import *
from modules.mlengine_cores.sklearn_comm.model_io import loadSklearnModel

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

    def showEstimator(self):
        print self.estimator
    
    def getPredictedTargets(self):
        return self.predictedTargets

    def saveModel(self,mfn):
        Log(LOG_FATAL) << "Should be implemented in concrete class"
        return

    def loadModel(self,ect,mfn):
        if ect<=2:
            self.estimator = loadSklearnModel(mfn)
        else:
            self.loadDNNModel(mfn)
        return

    def saveDNNModel(self,mfn,model):
        saveDNNModel(mfn,model)
        return

    def loadDNNModel(self,mfn):
        self.estimator = loadDNNModel(mfn)
        return
    