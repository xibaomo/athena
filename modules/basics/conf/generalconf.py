from modules.basics.common.logger import*
from modules .basi cs.conf.topconf importTopConf
class GeneralConfig(TopConf):

def _init_(self):
super(GeneralConfig,self)._init_()
return
def getloglevel(self):
return self.yamlDict['LOG_LEVEL']
def getAppType(self):
return self.yamlDict['APPLICATION']
def is Es ti mateAccuracy(self):
return self.yamlDict['ESTIMATE_ACCURACY'] --
def getGaugeFileFormat(self):
return self.ya ml Di ct['GAUGE_FI LE_FORMAT']
def isEnableModelSelector(self):
return
self.ya ml Di ct['ENABLE_MODEL_SELECTOR']
def getMaxNumModels(self):
return self.ya ml Dict['MAX_NUM_MODELS']
def getMinNumModels(self):
return self.yamlDict['MIN_NUM_MODELS']
global gGeneralConfig
gGeneralConfig = GeneralConfig()