'''
Created on Oct 1, 2018

@author: fxua
'''
from modules.feature_extractor.extractor import FeatureExtractor
from modules.basics.conf.topconf import TopConf
class FoxFexConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self,foxconfig):
        '''
        Constructor
        '''
        super(FoxFexConfig,self).__init__()
        self.loadYamlDict(foxconfig.getYamlDict()['FEATURE_EXTRACTOR'])
        return

    
    def getFastPeriod(self):
        return self.yamlDict['FAST_PERIOD']
    
    def getSlowPeriod(self):
        return self.yamlDict['SLOW_PERIOD']
    
    def getTickFile(self):
        return self.yamlDict['TICK_FILE']
    
    def getTrainSize(self):
        return self.yamlDict['TRAIN_SIZE']
    
    def getTestSize(self):
        return self.yamlDict['TEST_SIZE']
    
    def getFeatureList(self):
        return self.yamlDict['FEATURE_LIST']
    
    def getPositionType(self):
        return self.yamlDict['POSITION_TYPE']
        