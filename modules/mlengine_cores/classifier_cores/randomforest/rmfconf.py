'''
Created on Sep 4, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
class RMFConfig(TopConf):
    def __init__(self):
        super(RMFConfig,self).__init__()
        return
    
    def getNEstimator(self):
        return self.yamlDict['N_ESTIMATORS']
    
    def getCriterion(self):
        return self.yamlDict['CRITERION']
    
    def getMinSampleSplit(self):
        return self.yamlDict['MIN_SAMPLES_SPLIT']