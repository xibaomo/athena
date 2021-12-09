'''
Created on Oct 22, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
from modules.basics.conf.masterconf import gMasterConfig

class GaConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(GaConfig,self).__init__()
        self.loadYamlDict(gMasterConfig.getTotalYamlTree()['GENERIC_APP'])
        return
    
    def getEngineType(self):
        return self.yamlDict['ENGINE_TYPE']
    
    def getTrainDataFile(self):
        return self.yamlDict['TRAIN_DATA_FILES']

    def getVerifyDataFile(self):
        return self.yamlDict['VERIFY_DATA_FILES']

    def getTestRatio(self):
        return self.yamlDict['TEST_RATIO']

    def getShuffleRandomSeed(self):
        return self.yamlDict['SHUFFLE_RANDOM_SEED']

    def isSplitShuffle(self):
        return self.yamlDict['ENABLE_SPLIT_SHUFFLE']

    def getTargetHeader(self):
        return self.yamlDict['TARGET_HEADER']
    
    def getFeatureHeaders(self):
        return self.yamlDict['FEATURE_HEADERS']

    def getResultFileName(self):
        return self.yamlDict['RESULT_FILE']

    def isTestWholeSet(self):
        return self.yamlDict['TEST_WHOLE_SET']

    def isEvaluateModel(self):
        return self.yamlDict['EVALUATE_MODEL']
    
    