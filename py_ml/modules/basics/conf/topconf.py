'''

This file defines interfaces for various confs.
All confs except masterconf are derived from this
Created on Sep 3, 2018

@author: xfb
'''

class TopConf(object):
    def __init__(self):
        self.yamlDict={}
        return
    
    def loadYamlDict(self,yamlDict):
        self.yamlDict.update(yamlDict)
        return
    
    def getYamlDict(self):
        return self.yamlDict