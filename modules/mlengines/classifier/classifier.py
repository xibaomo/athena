'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.mlengines.mlengine import MLEngine
from modules.basics.common.logger import *
from sklearn.metrics import accuracy_score
import numpy as np

class Classifier(MLEngine):
    def __init__(self,engineCore=None):
        super(Classifier,self).__init__(engineCore)
        return
    
    def predict(self,fm):
        Log(LOG_DEBUG) << "Start predicting ..."
        self.testFeatureMatrix = fm
        self.engineCore.predict(fm)
        self.predicted_labels = self.engineCore.getPredictedTargets()
        
        Log(LOG_DEBUG) << "Prediction done"

        return
    
    def evaluatePrediction(self,fm,trueAns):
        labels = trueAns
        
        if labels is None:
            Log(LOG_INFO) << "No true labels found for test set, cannot evaluate prediction"
            return None,None
        
        feature_matrix = fm
        if len(labels) > 0:
            self._estimateAccuracy(labels)
        else:
            Log(LOG_FATAL) << "No correct labels are given for accuracy estimation"
            
        #extract failed feature and labels
        outlier_fm = []
        outlier_label = []
        
        for i in range(len(labels)):
            if labels[i] != self.predicted_labels[i]:
                outlier_fm.append(feature_matrix[i,:])                
                outlier_label.append(labels[i])
                
        if len(outlier_label) == 0:
            return [],[]
        
        return np.vstack(outlier_fm),np.array(outlier_label)
    
    def _estimateAccuracy(self,labels):
        acc = accuracy_score(labels, self.predicted_labels)
        Log(LOG_INFO) << "Model accuracy: %.3f " % acc 
        return