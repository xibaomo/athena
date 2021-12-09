'''
Created on Oct 1, 2018

@author: fxua
'''
import os
from modules.basics.conf.masterconf import gMasterConfig
from modules.basics.conf.topconf import TopConf
class FxtConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(FxtConfig,self).__init__()
        self.loadYamlDict(gMasterConfig.getTotalYamlTree()['FXT'])
        tickFile = self.yamlDict['FEATURE_EXTRACTOR']['TICK_FILE']
        self.modelFilePrefix,_ = os.path.splitext(tickFile);
        self.modelFilePrefix = self.modelFilePrefix.lower()
        return 
    
    def getPointValue(self):
        return self.yamlDict['POINT_VALUE']
        
    def getTrainerType(self):
        return self.yamlDict['TRAINER_TYPE']
    
    def getTakeProfit(self):
        return self.yamlDict['TAKE_PROFIT_POINTS']
    
    def getStopLoss(self):
        return self.yamlDict['STOP_LOSS_POINTS']

    def getModelPrefix(self):
        return self.modelFilePrefix
