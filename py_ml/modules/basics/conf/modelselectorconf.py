'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
from modules.model_selector.conf.dctconf import MS_DctConfig
from modules.model_selector.conf.rmfconf import MS_RmfConfig
from modules.model_selector.conf.dnnconf import MS_DNNConfig

class ModelSelectorConf(TopConf):
    def __init__(self):
        super(ModelSelectorConf,self).__init__()
        return
    
    def loadYamlDict(self, yamlDict):
        super(ModelSelectorConf,self).loadYamlDict(yamlDict)
        self.createModelList()
        self.cvFold = self.yamlDict['CV_FOLD']
        
        
    def createModelList(self):
        self.models=[]
        for d in self.yamlDict.keys():
            if d.startswith("MODEL_") and self.yamlDict[d]['ENABLED']:
                self.models.append(self.yamlDict[d])
                if self.yamlDict[d]['MODEL_NAME'] == 'DCT':
                    self.dctConfig = MS_DctConfig.getInstance()
                    self.dctConfig.loadYamlDict(self.yamlDict[d])
                if self.yamlDict[d]['MODEL_NAME'] == 'RMF':
                    self.rmfConfig = MS_RmfConfig.getInstance()
                    self.rmfConfig.loadYamlDict(self.yamlDict[d])
                if self.yamlDict[d]['MODEL_NAME'] == 'DNN':
                    self.dnnConfig = MS_DNNConfig.getInstance()
                    self.dnnConfig.loadYamlDict(self.yamlDict[d])
                    
    def getModelList(self):
        return self.models 
    
    def getCVFold(self):
        return self.cvFold
    
    def setCVFold(self,fold):
        self.cvFold = int(fold)
        
    def getNJobs(self):
        return self.yamlDict['N_JOBS']
    
    def getDCTConfig(self):
        return self.dctConfig
    
    def getRMFConfig(self):
        return self.rmfConfig
    
    def getDNNConfig(self):
        return self.dnnConfig
    
    def getSaveModelPrefix(self):
        return self.yamlDict['SAVE_MODEL_PREFIX']
    
    def getLoadModelPrefix(self):
        return self.yamlDict['LOAD_MODEL_PREFIX']
    
global gModelSelectorConfig
gModelSelectorConfig = ModelSelectorConf()