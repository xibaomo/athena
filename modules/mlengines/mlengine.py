'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.common.logger import *
from modules.basics.conf.generalconf import gGeneralConfig
from modules.basics.conf.mlengineconf import gMLEngineConfig
import numpy as np

class MLEngine(object):
    def __init__(self,engineCore=None):
        self.engineCore = engineCore
        self.predicted_labels = None
        return
    
    def loadEngineCore(self,engine_core):
        self.engineCore = engine_core
        return
    
    def train(self):
        Log(LOG_INFO) << "Start training model ..."
        self.engineCore.train()
        Log(LOG_INFO) << "Training is finished"
        
        return
    
    def predict(self):
        Log(LOG_FATAL) << "This function should be implemented in concrete class"
        return
    

    

        
