'''
Created on Oct 1, 2018

@author: fxua
'''
from apps.app import App
from apps.forex_trainer.fxtconf import FxtConfig
from apps.forex_trainer.fextor.forexfextor import ForexFextor
from modules.appframeworks.cascadedfilters.overkillfilters.overkillfilters import OverkillFilters
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
        self.fextor = ForexFextor(self.config)
        self.workForce = OverkillFilters(self.fextor)
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):
        self.fextor.loadTickFile()
        self.fextor.computeFeatures()
        return
    
    def execute(self):
        self.workForce.train()
        
        self.workForce.filterBadPoints()
        return
    
    def finish(self):
        return