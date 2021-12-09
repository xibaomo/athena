'''
Created on Oct 1, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
class ForexFexConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self,fxtconfig):
        '''
        Constructor
        '''
        super(ForexFexConfig,self).__init__()
        self.loadYamlDict(fxtconfig.getYamlDict()['FEATURE_EXTRACTOR'])
        return

    
    def getFastPeriod(self):
        return self.yamlDict['FAST_PERIOD']
    
    def getSlowPeriod(self):
        return self.yamlDict['SLOW_PERIOD']
    
    def getTickFile(self):
        return self.yamlDict['TICK_FILE']
    
#     def getTestPeriod(self):
#         return self.yamlDict['TEST_PERIOD']
    
    def getTestSize(self):
        return self.yamlDict['TEST_SIZE']
    
    def getFeatureList(self):
        return self.yamlDict['FEATURE_LIST']
    
    def getSignalPeriod(self):
        return self.yamlDict['SIGNAL_PERIOD']
    
    def getSampleRate(self):
        return self.yamlDict['SAMPLE_RATE']
    
    def getRSIPeriod(self):
        return self.yamlDict['RSI_PERIOD']
    
 
        