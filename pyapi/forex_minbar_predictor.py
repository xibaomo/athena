'''
Created on Dec 6, 2018

@author: fxua
'''
from apps.forex_bar_trainer.bar_feature_calculator import BarFeatureCalculator
from modules.mlengine_cores.sklearn_comm.model_io import loadSklearnModel
import numpy as np
import re
class ForexMinBarPredictor(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.featureCalculator =  BarFeatureCalculator()
        self.model = None
        return
    
    def loadAModel(self,modelFile):
        self.model = loadSklearnModel(modelFile)
        print self.model
        return
    
    def setFeatureNames(self,nameStr):
        self.featureNames = re.split('\W+',str(nameStr))
        print self.featureNames
        print "Num of features: %d" % len(self.featureNames)
        return
    
    def setLookback(self,lookback):
        self.featureCalculator.setLookback(lookback)
        print "Lookback: %d" % lookback
        return
    
    def classifyMinBar(self,open,high,low,close,tickvol):
        print "Predicting features: " + str(self.featureNames)
        self.featureCalculator.resetFeatureTable()
        self.featureCalculator.appendNewBar(open,high,low,close,tickvol)
        self.featureCalculator.computeFeatures(self.featureNames)
        features = self.featureCalculator.getLatestFeatures()
        
        print "Feature computed"
        nanList = np.where(np.isnan(features))[0]
        if len(nanList) >0:
            print "Nan found in features, skip ..."
            return 1
        
        features = features.reshape(1,-1)
        print "predicting features: " + str(features)
        
        pred = self.model.predict(features)
        
        print "prediction: %d" % pred
        
        return pred[0]
        
        
        
        
        
        
        
        