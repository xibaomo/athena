'''
Created on Sep 4, 2018

@author: fxua
'''
from apps.app import App
from modules.basics.common.logger import *
from modules.feature_extractor.wordcount.wordcounter import WordCounter
from modules.mlengines.classifier.classifier import Classifier
from modules.mlengine_cores.mlengine_core_creator import createMLEngineCore 
from modules.basics.conf.spmconf import gSPMConfig
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.basics.conf.modelselectorconf import gModelSelectorConfig
from modules.basics.conf.generalconf import gGeneralConfig

class SpamFilter(App):
    def __init__(self):
        super(SpamFilter,self).__init__()
        self.featureExtractor = WordCounter.getInstance()

        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):

        if gSPMConfig.getInputType() == 0:
            self.featureExtractor.setTrainDir(gSPMConfig.getTrainDataDir())
            self.featureExtractor.setTestDir(gSPMConfig.getTestDataDir())
        elif gSPMConfig.getInputType == 1:
            self.featureExtractor.setTrainFeatureFile(gSPMConfig.getTrainFeatureFile())
            self.featureExtractor.setTrainLabelFile(gSPMConfig.getTrainLabelFile())
        else:
            pass
        
        self.featureExtractor.prepare()
        
        Log(LOG_INFO) << "Extracting features from train set ..."
        self.featureExtractor.extractTrainFeatures()
        Log(LOG_INFO) << "done"
        
        fm = self.featureExtractor.getTrainFeatureMatrix()
        input_dim = fm.shape[1]
        
        engCore = createMLEngineCore(gMLEngineConfig.getEngineCoreType(),input_dim)
            
        self.mlEngine = Classifier(engCore)
        Log(LOG_INFO) << "App: spam filter is created"
        
        return
    
    def execute(self):

        
        Log(LOG_INFO) << "Extracting features from test set ..."
        self.featureExtractor.extractTestFeatures()
        Log(LOG_INFO) << "done"
        
        Log(LOG_INFO) << "Training ..."        
        fm = self.featureExtractor.getTrainFeatureMatrix()
        labels = self.featureExtractor.getTrainTargets()
        self.mlEngine.train(fm,labels)
        
        Log(LOG_INFO) << "Predicting ..."
        fm = self.featureExtractor.getTestFeatureMatrix()
        self.mlEngine.predict(fm)
        
        trueLabels = self.featureExtractor.getTestTargets()
        self.mlEngine.evaluatePrediction(fm,trueLabels)
        
        return
    
    def finish(self):
        return
            