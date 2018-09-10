'''
Created on Sep 9, 2018

@author: fxua
'''
import numpy as np
from modules.basics.common.logger import *
from modules.basics.conf.generalconf import gGeneralConfig
from modules.mlengines.classifier.classifier import Classifier
from modules.model_selector.modelselector import ModelSelector
from modules.basics.conf.modelselectorconf import gModelSelectorConfig

class OverkillFilters(object):
    '''
    This framework contains multiple filters to filter out bad points
    We intend to kill all the bad points, and can tolerate
    some of the good points that are mistakenly removed.
    This is why it's called "overkill"
    '''


    def __init__(self, fextor):
        '''
        Constructor
        '''
        self.featureExtractor = fextor
        self.productEngines = [] # used to store filters
        return
    
    @classmethod
    def getInstance(cls,fextor):
        return cls(fextor)
    
    def train(self):
        
        Log(LOG_INFO) << "Extracting train features ..."
        self.featureExtractor.extractTrainFeatures()
        Log(LOG_INFO) << "Train features are extracted"
        
        while(len(self.productEngines) < gGeneralConfig.getMinNumModels()):
            Log(LOG_INFO) << "Product engines are less than %d, start building ..." \
                            % gGeneralConfig.getMinNumModels()
            
            self.productEngines = []
            self.buildEngines()
            
        Log(LOG_INFO) << "Engines are built: %d" % len(self.productEngines)
        
        return
    
    def buildEngines(self):
        '''
        At each iteration, select best engine core by cross-validation, and then
        train with all the training data. Cache the engine to self.productEngines
        
        Then back test with all training data, only remove the true alarms, and enter the
        next iteration
        '''
        
        fm = self.featureExtractor.getTrainFeatureMatrix()
        labels = self.featureExtractor.getTrainTargets()
        
        for it in range(gGeneralConfig.getMaxNumModels()):
            self.featureExtractor.setTrainFeatureMatrix(fm)
            self.featureExtractor.setTrainTargets(labels)
            mlEngine = Classifier()
            modelSelector = ModelSelector(fm,labels)
            engCore = modelSelector.selectBestModel()
            mlEngine.loadEngineCore(engCore)
            
            mlEngine.train()
            
            self.productEngines.append(mlEngine)
            Log(LOG_INFO) << "Model %d out of %d is cached" % (it+1,gGeneralConfig.getMaxNumModels())
            
            #back test with train data
            
            self.featureExtractor.setTestFeatureMatrix(fm)
            self.featureExtractor.setTestTargets(labels)
            mlEngine.predict()
            _,failed_labels = mlEngine.evaluatePrediction()
            Log(LOG_INFO) << "Failed samples: %d " % len(failed_labels)
            
            predictedLabels = mlEngine.getPredictedTargets()
            newFM,newLabels = self.removeTrueAlarms(fm,labels,predictedLabels)
            
            numAlarms = sum(newLabels)
            if numAlarms < gModelSelectorConfig.getCVFold():
                if numAlarms <=1 :
                    Log(LOG_INFO) << "Only %d alarms left, CV stops" % numAlarms
                    break;
                Log(LOG_INFO) << "Only %d alarms left, change CV fold to %d" % (numAlarms,numAlarms)
                gModelSelectorConfig.setCVFold(numAlarms)
                
            fm = newFM
            labels = newLabels
            
        return
    
    def removeTrueAlarms(self,fm,labels,predictLabels):
        if not len(labels) == len(predictLabels):
            Log(LOG_FATAL) << "Labels from data is inconsistent with predicted label in length"
            
        resFM=[]
        resLabels=[]
        k=0
        for i in range(len(labels)):
            if labels[i] == predictLabels[i] and labels[i] > 0: #true alarms
                k+=1
                pass
            else:
                resFM.append(fm[i,:])
                resLabels.append(labels[i])
        
        if len(resLabels) == 0:
            return [],[]
        
        Log(LOG_INFO) << "Removed true alarms: %d" % k
        
        return np.vstack(resFM),np.array(resLabels)
        