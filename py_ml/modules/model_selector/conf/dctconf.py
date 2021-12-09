'''
Created on Sep 3, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
import numpy as np
from numpy import dtype 

class MS_DctConfig(TopConf):
    def __init__(self):
        super(MS_DctConfig,self).__init__()
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def getSplitList(self):
        params = self.yamlDict['SAMPLE_SPLIT']
        params = params.split(',')
        splits=np.linspace(int(params[0]),int(params[1]),int(params[2]),dtype=np.int32)
        
        return splits
        
        