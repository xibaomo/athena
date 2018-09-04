'''
Created on Sep 4, 2018

@author: fxua
'''
from apps.app import App
from modules.basics.common.logger import *
from modules.basics.conf.spmconf import gSPMConfig

class SpamFilter(App):
    def __init__(self):
        super(SpamFilter,self).__init__()
        Log(LOG_INFO) << "App: spam filter is created"
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):
        self.featureExtractor.setTestDir(gSPMConfig.getTestDataDir())
        if gSPMConfig.getInputType() == 0:
            self.featureExtractor.setTrainDir(gSPMConfig.getTrainDataDir())
            self.featureExtractor.setTestDir(gSPMConfig.getTestDataDir)
        elif gSPMConfig.getInputType == 1:
            self.featureExtractor.setTrainFeatureFile(gSPMConfig.getTrainFeatureFile())
            self.featureExtractor.setTrainLabelFile(gSPMConfig.getTrainLabelFile())
        else:
            pass
        
        self.featureExtractor.prepare()
        
        return
    
    def execute(self):
        self.mlEngine.train()
        self.mlEngine.predict()
        return
    
    def finish(self):
        return
            