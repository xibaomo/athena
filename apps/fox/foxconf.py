'''
Created on Oct 1, 2018

@author: fxua
'''
from modules.basics.conf.masterconf import gMasterConfig
from modules.basics.conf.topconf import TopConf
class FoxConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(FoxConfig,self).__init__()
        self.loadYamlDict(gMasterConfig.getTotalYamlTree()['FOX'])
        return 
        

        