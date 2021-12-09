'''
Created on Sep 4, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf

class DCTConfig(TopConf):
    def __init__(self):
        super(DCTConfig,self).__init__()
        return
    
    def getMinSampleSplit(self):
        return self.yamlDict['MIN_SAMPLE_SPLIT']
    
    def getCriterion(self):
        return self.yamlDict['CRITERION']