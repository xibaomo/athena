'''
Created on Oct 2, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
from modules.basics.conf.masterconf import gMasterConfig
class FFXConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.loadYamlDict(gMasterConfig.getTotalYamlTree()['FFX'])
        return
    
    def getTickFile(self):
        return self.yamlDict['TICK_FILE']
    
    def getFeatureList(self):
        return self.yamlDict['FEATURE_LIST']
    
    def getFastPeriod(self):
        return self.yamlDict['FAST_PERIOD']
    
    def getSlowPeriod(self):
        return self.yamlDict['SLOW_PERIOD']
    
    def getTestPeriod(self):
        return self.yamlDict['TEST_PERIOD']