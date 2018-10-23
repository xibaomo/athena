'''
Created on Oct 22, 2018

@author: fxua
'''

from modules.basics.conf.topconf import TopConf

class GaConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(GaConfig,self).__init__()
        return
    
    def getEngineType(self):
        return self.yamlDict['ENGINE_TYPE']
    
    def getDataFile(self):
        return self.yamlDict['DATA_FILE']
    
    def getTrainSize(self):
        return self.yamlDict['TRAIN_SIZE']
    
    def getTestSize(self):
        return self.yamlDict['TEST_SIZE']
         
    def getTargetHeader(self):
        return self.yamlDict['TARGET_HEADER']
    
    def getFeatureHeaders(self):
        return self.yamlDict['FEATURE_HEADERS']
    
    