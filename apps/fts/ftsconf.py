'''
Created on Oct 2, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
from modules.basics.conf.masterconf import gMasterConfig
from modules.forex_utils.common import FX_PointValue

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
        return self.yamlDict['TAKE_PROFIT_POINTS']
    
    def getStopLossPoint(self):
        return self.yamlDict['STOP_LOSS_POINTS']
    
    def getPointValue(self):
        return FX_PointValue[self.getFXSymbol()]
    
    def getSampleRate(self):
        return self.yamlDict['SAMPLE_RATE']
    
    def getExpiryPeriod(self):
        return self.yamlDict['EXPIRY_PERIOD']
    

    
