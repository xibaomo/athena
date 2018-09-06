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

class SpamFilter(App):
    def __init__(self):
        super(SpamFilter,self).__init__()
        self.featureExtractor = WordCounter.getInstance()
        engCore = createMLEngineCore(gMLEngineConfig.getEngineCoreType())
        self.mlEngine = Classifier(engCore)
        Log(LOG_INFO) << "App: spam filter is created"
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
        
        return
    
    def execute(self):
        Log(LOG_INFO) << "Extracting features from train set ..."
        x_train,y_train = self.featureExtractor.extractTrainFeatures()
        Log(LOG_INFO) << "done"
        
        Log(LOG_INFO) << "Extracting features from test set ..."
        x_test,y_test = self.featureExtractor.extractTestFeatures()
        Log(LOG_INFO) << "done"
        
        Log(LOG_INFO) << "Training ..."        
        self.mlEngine.train(x_train,y_train)
        
        Log(LOG_INFO) << "Predicting ..."
        self.mlEngine.predict(x_test,y_test)
        return
    
    def finish(self):
        return
            