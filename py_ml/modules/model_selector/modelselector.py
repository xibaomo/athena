'''
Created on Sep 5, 2018

@author: fxua
'''

from modules.basics.conf.modelselectorconf import gModelSelectorConfig
from modules.model_selector.svm_model_assessor import SVMModelAssessor
from modules.model_selector.dct_model_assessor import DCTModelAssessor
from modules.model_selector.rmf_model_assessor import RMFModelAssessor
from modules.basics.common.logger import *

class ModelSelector(object):
    def __init__(self,feature_matrix,labels):
        self.feature_matrix = feature_matrix
        self.labels = labels 
        return
    
    def buildAModel(self,model,selectedModels):
        if model['MODEL_NAME'] == 'SVM':
            Log(LOG_INFO) << "Building SVM models ..."
            svm_assessor = SVMModelAssessor(self.feature_matrix,self.labels)
            kernelList = model['KERNEL_LIST']
            
            for ker in kernelList:
                m = svm_assessor.assess(ker)
                selectedModels.append(m)
            
            return
        
        if model['MODEL_NAME'] == 'DCT':
            Log(LOG_INFO) << "Building DCT models ..."
            dct_assessor = DCTModelAssessor(self.feature_matrix,self.labels)
            m = dct_assessor.assess()
            
        if model['MODEL_NAME'] == 'RMF':
            Log(LOG_INFO) << "Building RMF models ..."
            rmf_assessor = RMFModelAssessor(self.feature_matrix,self.labels)
            m = rmf_assessor.assess()
            
        selectedModels.append(m)
        
        return
    
    def selectBestModel(self):
        selectedModels = []
        model_list = gModelSelectorConfig.getModelList()
        
        for model in model_list:
            self.buildAModel(model, selectedModels)
            
        #pick the winner
        bestScore = 0.
        bestModel = None
        for m in selectedModels:
            if m[1] > bestScore:
                bestScore = m[1]
                bestModel = m[0]
        
        Log(LOG_INFO) << "Best score is {} given by {}".format(bestScore,
                                                               bestModel.getEstimator())
        
        return bestModel
            
        
            
            
            
            
            