'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
class MLEngineConfig(TopConf):
    def __init__(self):
        
        super(MLEngineConfig,self).__init__()
        return
    
    def getEngineCoreType(self):
        return self.yamlDict['ENGINE_CORE_TYPE']
       
global gMLEngineConfig
gMLEngineConfig = MLEngineConfig()