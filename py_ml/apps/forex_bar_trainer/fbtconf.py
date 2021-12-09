'''
Created on Dec 2, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
from modules.basics.conf.masterconf import gMasterConfig
from apps.forex_bar_trainer.bar_feature_calculator import FEATURE_SIZE_DICT
from modules.basics.common.logger import *
from libpasteurize.fixes.feature_base import Feature

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
    
    def getMALookBack(self):
        return self.yamlDict['MA_LOOK_BACK']
    
    def getFeatureList(self):
        return self.getRegFeatures() + self.getLabelFeatures()
    
    def getRegFeatures(self):
        return self.yamlDict['REG_FEATURES']
    
    def getLabelFeatures(self):
        return self.yamlDict['LABEL_FEATURES']
    
    def getTestSize(self):
        return self.yamlDict['TEST_SIZE']
    
    def getProfitLoss(self):
        return self.yamlDict['PROFIT_LOSS']
    
    def getPosType(self):
        return self.yamlDict['POS_TYPE']
    
    def getForexSymbol(self):
        return self.yamlDict['FOREX_SYMBOL']
    
    def getRegFeatureSize(self):
        if len(FEATURE_SIZE_DICT) == 0:
            Log(LOG_FATAL) << "No feature is computed"
        
        fs=0
        for k in self.getRegFeatures():
            fs += FEATURE_SIZE_DICT[k]
            
        return fs
            
    
    