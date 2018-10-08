'''
Created on Oct 1, 2018

@author: fxua
'''
from apps.app import App
from apps.forex_trainer.fxtconf import FoxConfig
from apps.forex_trainer.fextor.forexfextor import ForexFextor
from modules.basics.conf.masterconf import gMasterConfig
class ForexMultiFilters(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexMultiFilters,self).__init__()
        self.config = FoxConfig()
        self.config.loadYamlDict(gMasterConfig.getTotalYamlTree()['FXT'])
        self.fextor = ForexFextor(self.config)
        return