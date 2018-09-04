'''
Created on Sep 4, 2018

@author: fxua
'''
from modules.mlengine_cores.mlengine_core import MLEngineCore
from sklearn.ensemble import RandomForestClassifier
from modules.mlengine_cores.classifier_cores.randomforest.rmfconf import RMFConfig
from modules.basics.common.logger import *
from modules.basics.conf.mlengineconf import gMLEngineConfig

class RandomForest(MLEngineCore):
    
    def __init__(self,est=None):
        super(RandomForest,self).__init__(est)
        
        if est is None:
            self.rmfConfig = RMFConfig()
            self.rmfConfig.loadYamlDict(gMLEngineConfig.getYamlDict()['RMF'])
            self.estimator = RandomForestClassifier(n_estimators=self.rmfConfig.getNEstimator(),
                                                    min_samples_split=self.rmfConfig.getMinSampleSplit(),
                                                    criterion=self.rmfConfig.getCriterion())
            Log(LOG_INFO) << "Classifier: random forest is created"
        else:
            self.estimator = est
                
        return
    
    def train(self,feature_matrix,labels):
        self.estimator.fit(feature_matrix,labels)
        return
    
    def predict(self,feature_matrix):
        self.predicted_labels = self.estimator.predict(feature_matrix)
        return
    
    def getPredictedLabels(self):
        return self.predicted_labels
        