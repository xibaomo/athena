'''
Created on Oct 1, 2018

@author: fxua
'''
from apps.app import App
from apps.fox.foxconf import FoxConfig
from apps.fox.fextor.foxfextor import FoxFextor
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
        self.fextor = FoxFextor(self.config)
        return