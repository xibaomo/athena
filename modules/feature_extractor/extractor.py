'''
Created on Sep 3, 2018

@author: fxua
'''

import gzip
import cPickle

from modules.basics.common.logger import *
import numpy as np
from modules.feature_extractor.wordcount.wordcounter import WordCounter

ExtractorSwitcher = {
    99: WordCounter.getInstance
    }

def createFeatureExtractor(extractorType):
    func = ExtractorSwitcher[extractorType]
    return func()


class FeatureExtractor(object):
    def __init__(self):
        return
    
    def save(self,filename,model):
        stream=gzip.open(filename,"wb")
        cPickle.dump(model,stream)
        stream.close()
        return
    
    def load(self,filename):
        stream = gzip.open(filename)
        model = cPickle.load(stream)
        stream.close()
        return model
    
    def getTrainFeatureMatrix(self):
        return self.feature_matrix
    
    def getTrainLabels(self):
        return self.labels
    
    def prepare(self,args=None):
        return
    
    def extractTrainFeatures(self):
        Log(LOG_FATAL) << "Should be implemented in concrete extractors"
        return
    
    def extractTestFeatures(self):
        Log(LOG_FATAL) << "Should be implemented in concrete extractors"
        return
    
