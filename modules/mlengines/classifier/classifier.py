'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.mlengines.mlengine import MLEngine
from modules.basics.common.logger import *
import numpy as np

class Classifier(MLEngine):
    def __init__(self,engineCore=None):
        super(Classifier,self).__init__(engineCore)
        return
    
    def predict(self,feature_matrix,labels=None):
        Log(LOG_INFO) << "Start predicting ..."
        self.engineCore.predict(feature_matrix)
        self.predicted_labels = self.engineCore.getPredictedLabels()
        
        Log(LOG_INFO) << "Prediction done"
        
        if labels is None:
            return None,None
        
        if len(labels) > 0:
            self.estimateAccuracy(labels)
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