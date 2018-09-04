'''
Created on Sep 3, 2018

@author: fxua
'''
import os
import numpy as np
from collections import Counter
from modules.feature_extractor.extractor import FeatureExtractor
from modules.basics.common.logger import *
from modules.basics.conf.spmconf import gSPMConfig
from mhlib import Folder

NUM_WORDS = 3000

class WordCounter(FeatureExtractor):
    def __init__(self):
        super(WordCounter,self).__init__()
        self.trainDir = ""
        self.testDir = ""
        self.trainFEatureFile = ""
        self.trainLabelFile = ""
        Log(LOG_INFO) << "Feature_extractor: word counter is created"
        
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def setTrainDir(self,foldername):
        self.trainDir = foldername
        return
    
    def setTestDir(self,fd):
        self.testDir = fd
        return
    
    def setTrainFeatureFile(self,fn):
        self.trainFEatureFile = fn
        return
    
    def setTrainLabelFile(self,fn):
        self.trainLabelFile = fn
        return
    
    def prepare(self, args=None):
        self.makeDictionary(self.trainDir)
        return
    
    def makeDictionary(self,foldername):
        all_words = []
        mails = [os.path.join(foldername,f) for f in os.listdir(foldername)]
        for mail in mails:
            with open(mail) as  m:
                for line in m:
                    words = line.split()
                    all_words +=words
        dictionary = Counter(all_words)
        list_to_remove = dictionary.keys()
        
        for item in list_to_remove:
            if item.isalpha() == False:
                del dictionary[item]
            elif len(item) == 1:
                del dictionary[item]
            else:
                pass
            
        self.dictionary = dictionary.most_common(NUM_WORDS)
        
        return self.dictionary
    
    def extractFeatureLabel(self,foldername):
        if foldername == "":
            Log(LOG_FATAL) << "folder to extract is missing"
            
        files = [os.path.join(foldername,fi) for fi in os.listdir(foldername)]
        feature_matrix = np.zeros((len(files),NUM_WORDS))
        labels = np.zeros(len(files))
        
        count = 0
        docID = 0
        for fil in files:
            with open(fil) as fi:
                for i,line in enumerate(fi):
                    if i==2:
                        words = line.split()
                        for word in words:
                            wordID = 0
                            for i,d in enumerate(self.dictionary):
                                if d[0] == word:
                                    wordID = i 
                                    feature_matrix[docID,wordID] = words.count(word)
                labels[docID] = 0
                filepathTokens = fil.split('/')
                lastToken = filepathTokens[len(filepathTokens)-1]
                if lastToken.starts_with("spmsg"):
                    labels[docID] = 1
                    count = count+1
                docID = docID + 1
                
    def extractTrainFeatures(self):
        if self.trainDir == "":
            feature_matrix = self.load(gSPMConfig.getTrainFeatureFile())
            labels = self.load(gSPMConfig.getTrainLabelFile())
            
            return feature_matrix,labels
        
        foldername = self.trainDir 
        Log(LOG_INFO) << "Extracting features from " + foldername
        
        feature_matrix,labels = self.extractFeatureLabel(foldername)
        
        self.save(foldername+"_features.dat", feature_matrix)
        self.save(foldername+"_labels.dat", labels)
        
        Log(LOG_INFO) << "Dumped feature and label data files of training set"
        
        return feature_matrix,labels
    
    def extractTestFeatures(self):
        foldername = self.testDir
        Log(LOG_INFO) << "Extracting features from " + foldername
        fm,la = self.extractFeatureLabel(foldername)
        return fm,la
        
        
                    
                
                
    