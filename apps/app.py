'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *

class App(object):
    def __init__(self):
        return
    
    def prepare(self):
        Log(LOG_FATAL) << "Should be implemented in concrete app"
        return
    
    def execute(self):
        Log(LOG_FATAL) << "Should be implemented in concrete app"
        return
    
    def finish(self):
        Log(LOG_FATAL) << "Should be implemented in concrete app"
        return