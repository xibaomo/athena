'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
class FexConfig(TopConf):
    def __init__(self):
        super(FexConfig,self).__init__()
        return
    
    def getExtractorType(self):
        return self.yamlDict.get("EXTRACTOR_TYPE")
    
    def getCrossValidTestSize(self):
        tmpDict = self.yamlDict['CROSS_VALIADATION']
        return tmpDict.get('TEST_SIZE')
    
    def isCrossValid(self):
        if self.getCrossValidTestSize() == None:
            return False
        
        return True
    
global gFexConfig
gFexConfig = FexConfig()