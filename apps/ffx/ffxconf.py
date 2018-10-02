'''
Created on Oct 2, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
from modules.basics.conf.masterconf import gMasterConfig
FX_PointValue = {
    'EURUSD': 0.00001,
    'GBPUSD': 0.00001,
    'USDCHF': 0.00001,
    'USDJPY': 0.001,
    'USDCAD': 0.00001,
    'AUDUSD': 0.00001,
    'EURCHF': 0.00001,
    'EURJPY': 0.001,
    'EURGBP': 0.00001,
    'EURCAD': 0.00001,
    'GBPCHF': 0.00001,
    'GBPJPY': 0.001,
    'AUDJPY': 0.001
    }

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
    
    def getFXSymbol(self):
        return self.yamlDict['FOREX_SYMBOL']
    
    def getTakeProfitPoint(self):
        return self.yamlDict['TAKE_PROFIT_POINT']
    
    def getStopLossPoint(self):
        return self.yamlDict['STOP_LOSS_POINT']
    
    def getPointValue(self):
        return FX_PointValue[self.getFXSymbol()]
    
    def getPositionLife(self):
        return self.yamlDict['POSITION_LIFE']
    
    
    