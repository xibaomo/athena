'''
Created on Dec 6, 2018

@author: fxua
'''
from apps.forex_bar_trainer.bar_feature_calculator import BarFeatureCalculator,\
    GLOBAL_PROB_PERIOD
from modules.mlengine_cores.sklearn_comm.model_io import loadSklearnModel
import numpy as np
import re
import pandas as pd

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
        self.isBinom = False
        return
    
    def setInitMin(self,minbar,barfile):
        self.featureCalculator.setInitMin(minbar)
        latest_time = self.loadHistoryBarFile(barfile)
        
        print "Latest time in history after setting init_time: " + latest_time
        return
    
    def loadAModel(self,modelFile):
        self.model = loadSklearnModel(modelFile)
        print self.model
        return
    
    def loadHistoryBarFile(self,barFile):
        latest_time = self.featureCalculator.loadMinBars(barFile)
        
        return latest_time
    
    def loadHistoryMinBars(self,data,histLen,minbar_size):
        
        print "loading received minbars"
        data = np.array(data).reshape(histLen,minbar_size)
        
        print "first: ", data[:5,:]
        print "last: ", data[-5:,:]
        
#         k=0
#         while k < data.shape[0]-1:
#             self.featureCalculator.appendNewBar(data[k,0],data[k,1],data[k,2],
#                                                 data[k,3],data[k,4])
#             k+=1

        self.featureCalculator.loadHistoryMinBars(data[:-1,:])
        
        self.featureCalculator.markUnlabeled(tp=200, spread=10, digits=1.0e-5)
        
        if "BINOM" not in self.featureNames:
            return 
                
        self.predictHistoryMinBars()  
        return
    
    def setFeatureNames(self,nameStr):
        self.featureNames = re.split('\W+',str(nameStr))
        if "BINOM" in self.featureNames:
            self.isBinom = True
            
        print self.featureNames
        print "Num of features: %d" % len(self.featureNames)
        return
    
    def setLookback(self,lookback):
        self.featureCalculator.setLookback(lookback)
        self.lookback=lookback
        print "Lookback: %d" % lookback
        return
    
    def setMALookback(self,malookback):
        self.featureCalculator.setMALookback(malookback)
        print "MA_lookback: %d" % malookback
        return
    
    def classifyMinBar(self,open,high,low,close,tickvol,timestr):
        print "Predicting features: " + str(self.featureNames)
        self.featureCalculator.resetFeatureTable()
        
        print "append new bar: %f, %f, %f, %f, %d" % (open,high,low,close,tickvol)
        self.featureCalculator.appendNewBar(open,high,low,close,tickvol,timestr)
        
        print "Correcting past labels ..."
        self.featureCalculator.correctPastPrediction()
        
        if 'BINOM' in self.featureNames:
            self.featureNames.remove('BINOM')
            
        self.featureCalculator.computeFeatures(self.featureNames)
        features = self.featureCalculator.getLatestFeatures()
        
        if self.isBinom is True:
            print "Compute latest binom ..."
            sells,pb = self.featureCalculator.compLatestBinom()
            features = np.append(features, [sells,pb])
        
        nanList = np.where(np.isnan(features))[0]
        if len(nanList) >0:
            print "Nan found in features, skip ..."
            print features
            return 1
        
        features = features.reshape(1,-1)
        print "predicting features: " + str(features)
        
        pred = self.model.predict(features)
        
        self.featureCalculator.appendLatestLabel(pred[0])
        
        print "prediction: %d" % pred
        
        return pred[0]
        
    def predictHistoryMinBars(self):
        print "predicting all history bars ..."
        
        self.featureCalculator.computeFeatures(self.featureNames,GLOBAL_PROB_PERIOD*3)
        
        print "all features ready"
        self.featureCalculator.predictUnlabeledBars(self.model,self.lookback)
        
        print "All history bars labeled"
        return
        
        
        
        
        