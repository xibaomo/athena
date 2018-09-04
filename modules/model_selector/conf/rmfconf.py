'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
import numpy as np
class MS_RmfConfig(TopConf):
    def __init__(self):
        super(MS_RmfConfig,self).__init__()
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def getNEstimators(self):
        pms = self.yamlDict['N_ESTIMATORS']
        pms = pms.split(',')
        n_est = np.linspace(int(pms[0]), int(pms[1]), int(pms[2]), dtype=np.int64)
        
        return n_est
    
    def getMinSamplesSplit(self):
        pms =  self.yamlDict['MIN_SAMPLES_SPLIT']
        pms = pms.split(',')
        splits = np.linspace(int(pms[0]), int(pms[1]), int(pms[2]), dtype=np.int64)
        
        return splits
    
    def getCriterion(self):
        return self.yamlDict['CRITERION']