'''
Created on Sep 4, 2018

@author: fxua
'''
import ast
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
    
    def getMinSamplesLeaf(self):
        return self.yamlDict['MIN_SAMPLES_LEAF']
    
    def getClassWeight(self):
        val= self.yamlDict['CLASS_WEIGHT']
        if val[0] == '{':
            val = ast.literal_eval(val)
        
        return val
        
    def getNJobs(self):
        return self.yamlDict['N_JOBS']
    
    def getMaxDepth(self):
        val = self.yamlDict['MAX_DEPTH']
        if val == "":
            return None
        
        return val