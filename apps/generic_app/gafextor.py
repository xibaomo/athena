'''
Created on Oct 22, 2018

@author: fxua
'''
from modules.feature_extractor.extractor import FeatureExtractor
import pandas as pd
from modules.basics.common.logger import *
from sklearn.preprocessing import LabelEncoder
class GaFextor(FeatureExtractor):
    '''
    classdocs
    '''


    def __init__(self,config):
        '''
        Constructor
        '''
        super(GaFextor,self).__init__()
        self.config = config
        self.encoder = LabelEncoder()
        return
    
    def prepare(self, args=None):
        self.extractTotalFeatures()
          
        return
    
    def extractTotalFeatures(self):
        df = pd.read_csv(self.config.getDataFile())
        
        if self.config.getTargetHeader() == "":
            alltar = df.values[:,-1]
            
        else:
            alltar = df[self.config.getTargetHeader].values
            
        self.encoder.fit(alltar)
        alltar = self.encoder.transform(alltar)
            
        fhs = self.config.getFeatureHeaders()
        if len(fhs) == 0:
            allfm = df.values[:,:-1]
        else:
            allfm = df[self.config.getFeatureHeaders()].values
        
        allfm = self.scaler.fit_transform(allfm)
        testSize = self.config.getTestSize()
        self.trainFeatureMatrix = allfm[:-testSize,:]
        self.testFeatureMatrix = allfm[:-testSize,:]
        self.trainTargets = alltar[:-testSize]
        self.testTargets = alltar[-testSize:]
        
        self.allfm = allfm
        self.alltar = alltar
        return
    
    def getTotalFeatureMatrix(self):
        return self.allfm
    
    def getTotalTargets(self):
        return self.alltar
        