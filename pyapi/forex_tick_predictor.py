'''
Created on Nov 16, 2018

@author: fxua
'''
from modules.forex_utils.feature_calculator import FeatureCalculator
from modules.mlengine_cores.sklearn_comm.model_io import loadSklearnModel
import numpy as np
class ForexTickPredictor(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.featureCalculator = FeatureCalculator()
        self.prodcutModels = []
        return
    
    def setPeriods(self,fast,slow):
        self.featureCalculator.setPeriods(fast, slow)
        return
    
    def loadTicks(self,ticks):
        self.featureCalculator.loadPrice(np.array(ticks))
        return
    
    def loadAModel(self,modelFile):
        model = loadSklearnModel(modelFile)
        self.prodcutModels.append(model)
        print model
        return
    
    def setFeatureNames(self,nameStr):
        self.featureNames = str(nameStr).split(',')
        return
    
    def classifyATick(self,tick):
        self.featureCalculator.appendPrice(tick)
        self.featureCalculator.computeFeatures(self.featureNames)
        features = self.featureCalculator.getLatestfeatures()
        
        nanList = np.where(np.isnan(features))[0]
        if len(nanList) > 0:
            print "Nan found in features, skip ..."
            return 1
        
        for m in self.prodcutModels:
            pred = m.predict(features)
            if pred == 1:
                return 1;
        
        return pred
        
        