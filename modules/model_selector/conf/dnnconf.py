'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
import numpy as np

class MS_DNNConfig(TopConf):
    def __init__(self):
        super(MS_DNNConfig,self).__init__()
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def getTrainSteps(self):
        return self.yamlDict['TRAIN_STEPS']
    
    def getLayerNodes(self):
        return self.yamlDict['LAYER_NODES']
    
    def getFittingIters(self):
        return self.yamlDict['FITTING_ITERS']