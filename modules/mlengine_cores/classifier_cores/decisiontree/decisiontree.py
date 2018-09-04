'''
Created on Sep 4, 2018

@author: fxua
'''
from modules.mlengine_cores.classifier_cores.decisiontree.dctconf import DCTConfig
from sklearn import tree 
from modules.mlengine_cores.mlengine_core import MLEngineCore
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.basics.common.logger import *

class DecisionTree(MLEngineCore):
    def __init__(self,est=None):
        super(DecisionTree,self).__init__(est)
        if est is None:
            self.dctConfig = DCTConfig()
            self.dctConfig.loadYamlDict(gMLEngineConfig.getYamlDict()['DCT'])
            self.estimator = tree.DecisionTreeClassifier(criterion=self.dctConfig.getCriterion(),
                                                         min_samples_split=self.dctConfig.getMinSampleSplit())
            Log(LOG_INFO) << "Classifier: Decision tree is created:\n" + \
                             "\t\t\t\t criterion: " + self.dctConfig.getCriterion() + "\n" + \
                             "\t\t\t\t min_samples_split: " + str(self.dctConfig.getMinSampleSplit())
        else:
            self.estimator = est
            
        return
    
    def train(self, feature_matrix, targets):
        self.estimator.fit(feature_matrix,targets)
        return
    
    def predict(self, feature_matrix):
        self.predicted_labels = self.estimator.predict(feature_matrix)
        return
    
    def getPredictedLabels(self):
        return self.predicted_labels
                             