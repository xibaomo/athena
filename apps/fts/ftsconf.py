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

class FTSConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(FTSConfig,self).__init__()
        self.loadYamlDict(gMasterConfig.getTotalYamlTree()['FTS'])
        return
    
    def getTickFile(self):
        return self.yamlDict['TICK_FILE']
    
    def getFXSymbol(self):
        return self.yamlDict['FOREX_SYMBOL']
    
    def getTakeProfitPoint(self):
        return self.yamlDict['TAKE_PROFIT_POINT']
    
    def getStopLossPoint(self):
        return self.yamlDict['STOP_LOSS_POINT']
    
    def getPointValue(self):
        return FX_PointValue[self.getFXSymbol()]
    
    def getSampleRate(self):
        return self.yamlDict['SAMPLE_RATE']
    
    def getExpiryPeriod(self):
        return self.yamlDict['EXPIRY_PERIOD']
    
    