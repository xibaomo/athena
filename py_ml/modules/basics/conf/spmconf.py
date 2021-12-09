'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *
from modules.basics.conf.topconf import TopConf

class SPMConfig(TopConf):
    def __init__(self):
        super(SPMConfig,self).__init__()
        return
    
    def getTrainDataDir(self):
        return self.yamlDict.get('TRAIN_DATA_DIR')
    
    def getTestDataDir(self):
        return self.yamlDict.get('TEST_DATA_DIR')
    
    def getTrainFeatureFile(self):
        return self.yamlDict.get('TRAIN_FEATURE_FILE')
    
    def getTrainLabelFile(self):
        return self.yamlDict.get('TRAIN_LABEL_FILE')
    
    def getTestFeatureFile(self):
        return self.yamlDict.get('TEST_FEATURE_FILE')
    
    def getInputType(self):
        return self.yamlDict.get('INPUT_TYPE')
    
global gSPMConfig
gSPMConfig = SPMConfig()
