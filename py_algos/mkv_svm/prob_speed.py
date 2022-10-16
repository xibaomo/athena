#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 03:53:55 2022

@author: naopc
"""
import pdb

class ProbSpeedConfig(object):
    def __init__(self,gencfg):
        self.yamlDict = gencfg.yamlDict
        self.root = "PROB_SPEED"
        
    def getProbNodes(self):
        return self.yamlDict[self.root]['PROB_NODES']
    
    def getMinSpeed(self):
        return self.yamlDict[self.root]['MIN_SPEED']
        
class ProbSpeedPredictor(object):
    def __init__(self,gencfg):
        self.cfg = ProbSpeedConfig(gencfg)
        
    def predict(self,fm):
        prob_buy = fm[0,0]
        spd = fm[0,1]
        
        if spd < self.cfg.getMinSpeed():
            print("Speed too low. No action. ",spd,self.cfg.getMinSpeed())
            return 0

        act = 0

        prob_low = self.cfg.getProbNodes()[0]
        prob_high = self.cfg.getProbNodes()[1]    
        
        if prob_buy >= prob_high:
            act = 2
        if prob_buy <= prob_low:
            act = 1
            
        return act