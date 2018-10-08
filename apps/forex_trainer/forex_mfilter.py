'''
Created on Oct 1, 2018

@author: fxua
'''
from apps.app import App
from apps.forex_trainer.fxtconf import FoxConfig
from apps.forex_trainer.fextor.forexfextor import ForexFextor
class ForexMultiFilter(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexMultiFilter,self).__init__()
        self.config = FoxConfig()
        self.fextor = ForexFextor(self.config)
        return