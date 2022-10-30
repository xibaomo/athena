#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 04:40:57 2022

@author: naopc
"""
import pickle
class PklPredictorConfig(object):
    def __init__(self,gencfg):
        self.yamlDict = gencfg.yamlDict
        self.root = 'PKL_PREDICTOR'
        
    def getMinSpeed(self):
        return self.yamlDict[self.root]['MIN_SPEED']
        
    def getModelFile(self):
        return self.yamlDict[self.root]['MODEL_FILE']
    
    def getScalerFile(self):
        return self.yamlDict[self.root]['SCALER_FILE']
    
    def getSelectedFeatureID(self):
        return self.yamlDict[self.root]['SELECTED_FEATURE_ID']
    
class PklPredictor(object):
    def __init__(self,gencfg):
        self.cfg = PklPredictorConfig(gencfg)
        self.model = pickle.load(open(self.cfg.getModelFile(),'rb'))
        self.scaler = pickle.load(open(self.cfg.getScalerFile(),'rb'))
        
    def predict(self,fm):
        spd = fm[0,1]
        if abs(spd) < self.cfg.getMinSpeed():
            print("Speed too low. No action. ",spd,self.cfg.getMinSpeed())
            return 0
        
        ft_id = self.cfg.getSelectedFeatureID()
        pfm = fm[:,ft_id] # must be a 2d array
        ffm = self.scaler.transform(pfm)
        act = self.model.predict(ffm)
        return act[0]