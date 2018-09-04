'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
class MLEngineConfig(TopConf):
    def __init__(self):
        
        super(MLEngineConfig,self).__init__()
        return
    
    def getEngineType(self):
        return self.yamlDict['ENGINE_TYPE']
    
    def getTrainSteps(self):
        return self.yamlDict['TRAIN_STEPS']
    
    
global gMLEngineConfig
gMLEngineConfig = MLEngineConfig()