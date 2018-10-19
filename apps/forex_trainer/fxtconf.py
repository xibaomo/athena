'''
Created on Oct 1, 2018

@author: fxua
'''
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
        return 
    
    def getPointValue(self):
        return self.yamlDict['POINT_VALUE']
        
    def getTrainerType(self):
        return self.yamlDict['TRAINER_TYPE']

        