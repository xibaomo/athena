'''
Created on Sep 4, 2018

@author: fxua
'''
from modules.mlengine_cores.mlengine_core import MLEngineCore
from sklearn.ensemble import RandomForestClassifier
from modules.mlengine_cores.classifier_cores.randomforest.rmfconf import RMFConfig
from modules.basics.common.logger import *
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.mlengine_cores.sklearn_comm.model_io import saveSklearnModel,\
    loadSklearnModel

class RandomForest(MLEngineCore):
    
    def __init__(self,est=None):
        super(RandomForest,self).__init__(est)
        
        if est is None:
            self.rmfConfig = RMFConfig()
            self.rmfConfig.loadYamlDict(gMLEngineConfig.getYamlDict()['RMF'])
            self.estimator = RandomForestClassifier(n_estimators=self.rmfConfig.getNEstimator(),
                                                    min_samples_split=self.rmfConfig.getMinSampleSplit(),
                                                    criterion=self.rmfConfig.getCriterion())
            Log(LOG_INFO) << "Classifier: random forest is created: {}".format(self.estimator)
        else:
            self.estimator = est
                
        return
    
    def saveModel(self, mfn):
        saveSklearnModel(mfn,self.estimator)
        return
    
    def loadModel(self, mfn):
        self.estimator = loadSklearnModel(mfn)
        return