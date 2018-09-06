'''
Created on Sep 3, 2018

@author: xfb
'''
import yaml
import os
from modules.basics.common.logger import *
from modules.basics.conf.generalconf import gGeneralConfig
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.basics.conf.spmconf import gSPMConfig
from modules.basics.conf.fexconf import gFexConfig
from modules.basics.conf.modelselectorconf import gModelSelectorConfig
import modules.basics.common.utils as utils

class MasterConfig(object):
    def __init__(self):
        self.loadGlobalConfig()
        return
        
        
    def loadGlobalConfig(self):
        globalConfigFile = os.environ["ATHENA_INSTALL"] + "/config/athena.yaml"
        self.yamlDict = yaml.load(open(globalConfigFile))
        return
    
    def loadLocalConfig(self,configFile):
        localYamlRoot = yaml.load(open(configFile))
        allLocalKeys=[]
        utils.getAllYamlKeys(localYamlRoot, "", allLocalKeys)
        
        for key in allLocalKeys:
            v = utils.findYamlKey(self.yamlDict, key)
            if v == None:
                Log(LOG_FATAL) << "Key not found %s" % key
                return
            v = utils.findYamlKey(localYamlRoot, key)
            utils.updateYamlKey(self.yamlDict, key, v)
            
        return
    
    def prepare(self):
        self.__divideYamlDict()
        return
    
    def __divideYamlDict(self):
        gGeneralConfig.loadYamlDict(self.yamlDict['GENERAL'])
        gMLEngineConfig.loadYamlDict(self.yamlDict['ML_ENGINE_CORE'])
        gSPMConfig.loadYamlDict(self.yamlDict['SPM'])
        gFexConfig.loadYamlDict(self.yamlDict['FEATURE_EXTRACTOR'])
        gModelSelectorConfig.loadYamlDict(self.yamlDict['MODEL_SELECTOR'])
        
        return
    
    def getTotalYamlTree(self):
        return self.yamlDict
    
global gMasterConfig
gMasterConfig = MasterConfig()
        