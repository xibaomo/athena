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
from modules.appframeworks.cascadedfilters.overkillfilters.overkillfilters import OverkillFilters
class SpamMultiFilters(App):
    def __init__(self):
        super(SpamMultiFilters,self).__init__()
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
        self.featureExtractor.extractTrainFeatures()
        
        self.workForce = OverkillFilters(self.featureExtractor)
        
        return
    
    def execute(self):
        self.workForce.train()
        
        self.workForce.filterBadPoints()
  
        return
    
    def finish(self):
        return
            