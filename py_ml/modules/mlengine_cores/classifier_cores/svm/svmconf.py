'''
Created on Sep 4, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf

class SVMConfig(TopConf):
    def __init__(self):
        super(SVMConfig,self).__init__()
        return
    
    def getKernel(self):
        return self.yamlDict['KERNEL']
    
    def getRegularParam(self):
        return self.yamlDict['C']
    
    def getGamma(self):
        return self.yamlDict['GAMMA']
    
    def getCoef0(self):
        return self.yamlDict['COEF0']
    
    def getDegree(self):
        return self.yamlDict['DEGREE']