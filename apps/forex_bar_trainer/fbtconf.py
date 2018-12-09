'''
Created on Dec 2, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
from modules.basics.conf.masterconf import gMasterConfig

class FbtConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(FbtConfig,self).__init__()
        self.loadYamlDict(gMasterConfig.getTotalYamlTree()['FOREX_BAR_FILTER'])
        return
    
    def getBarFile(self):
        return self.yamlDict['BAR_FILE']
    
    def getLookBack(self):
        return self.yamlDict['LOOK_BACK']
    
    def getFeatureList(self):
        return self.yamlDict['FEATURE_LIST']
    
    def getTestSize(self):
        return self.yamlDict['TEST_SIZE']
    
    def getProfitLoss(self):
        return self.yamlDict['PROFIT_LOSS']
    
    def getPosType(self):
        return self.yamlDict['POS_TYPE']
    
    def getForexSymbol(self):
        return self.yamlDict['FOREX_SYMBOL']
    
    
    