'''
Created on Oct 22, 2018

@author: fxua
'''
from modules.feature_extractor.extractor import FeatureExtractor
import pandas as pd
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
        return
    
    def prepare(self, args=None):
        self.extractTotalFeatures()
          
        return
    
    def extractTotalFeatures(self):
        df = pd.read_csv(self.config.getDataFile())
        alltar = df[self.config.getTargetHeader].values
        allfm = df[self.config.getFeatureHeaders()].values
        
        testSize = self.config.getTestSize()
        self.trainFeatureMatrix = allfm[:-testSize,:]
        self.testFeatureMatrix = allfm[:-testSize,:]
        self.trainTargets = alltar[:-testSize]
        self.testTargets = alltar[-testSize:]
        return
    
    
        