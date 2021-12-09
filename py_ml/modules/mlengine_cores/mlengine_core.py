'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *
from modules.mlengine_cores.dnncomm.dnncreator import *
from modules.mlengine_cores.sklearn_comm.model_io import loadSklearnModel
from sklearn.metrics import accuracy_score

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
        saveDNNModel(model,mfn)
        return

    def loadDNNModel(self,mfn):
        self.estimator = loadDNNModel(mfn)
        return
    
    def getAccuracy(self,fm,tar):
        self.predict(fm)
        pred = self.getPredictedTargets()
        acc = accuracy_score(tar,pred)
        miss=0
        crt=0
        for t,p in zip(tar,pred):
            if t==1 and p == 0:
                miss+=1
            if t==p:
                crt+=1
        miss_ratio = float(miss)/len(tar)
        return acc,miss_ratio,crt
    