'''
Created on Oct 1, 2018

@author: fxua
'''
from apps.app import App
from apps.forex_trainer.fxtconf import FxtConfig
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
        self.config = FxtConfig()
        self.config.loadYamlDict(gMasterConfig.getTotalYamlTree()['FXT'])
        self.fextor = ForexFextor(self.config)
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):
        self.fextor.loadTickFile()
        self.fextor.computeFeatures()
        return
    
    def execute(self):
        return
    
    def finish(self):
        return