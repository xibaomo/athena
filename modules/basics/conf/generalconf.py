from modules.basics.common.logger import *
from modules.basics.conf.topconf import TopConf

class GeneralConfig(TopConf):
    def __init__(self):
        super(GeneralConfig,self).__init__()
        return
    
    def getLogLevel(self):
        return self.yamlDict['LOG_LEVEL']
    
    def getAppType(self):
        return self.yamlDict['APPLICATION']
    
    def getGaugeFileFormat(self):
        return self.yamlDict['GAUGE_FILE_FORMAT']
    
    def isEnableModelSelector(self):
        return self.yamlDict['ENABLE_MODEL_SELECTOR']
    
    def getMaxNumModels(self):
        return self.yamlDict['MAX_NUM_MODELS']
    
    def getMinNumModels(self):
        return self.yamlDict['MIN_NUM_MODELS']
    
global gGeneralConfig
gGeneralConfig = GeneralConfig()