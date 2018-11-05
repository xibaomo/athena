'''
Created on Oct 22, 2018

@author: fxua
'''
from modules.feature_extractor.extractor import FeatureExtractor
import modules.basics.common.utils as utils
import numpy as np
import pandas as pd
import re
import os
from modules.basics.common.logger import *
from sklearn.model_selection import train_test_split
from modules.basics.conf.generalconf import gGeneralConfig
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

    def getFeatureHeaders(self):
        return self.headers

    def getFeatureFromFile(self,filename):
        df = pd.read_csv(filename)
        headers = self.config.getFeatureHeaders().strip()
        tg_header = self.config.getTargetHeader()
        if headers == "": # if no feature headers given, use all columns except target
            for key in df.keys():
                if not key == tg_header:
                    headers.append(key)
        else:
            headers = re.split('; |, |,',headers)
        self.headers = headers

        fm = df.loc[:,headers].values
        targets = df.get(tg_header,None)
        if targets is not None:
            targets = np.array(targets)

        targets = targets.reshape(-1,1)
        return fm,targets

    def extractTrainFeatures(self):
        trainFiles = self.config.getTrainDataFiles()

        allfm=[]
        alltar = []
        for tf in trainFiles:
            fm,tar = self.getFeatureFromFile(tf)
            allfm.append(fm)
            alltar.append(tar)

        self.totalFeatureMatrix = np.vstack(allfm)

        self.totalTargets = np.vstack(alltar)

        # Standardization
        if gGeneralConfig.isEnableFeatureStandardization():
            if not gGeneralConfig.getLoadScalerFile() == "":
                self.totalFeatureMatrix = self.scaler.transform(self.totalFeatureMatrix)
            else:
                self.totalFeatureMatrix = self.scaler.fit_transform(self.totalFeatureMatrix)

        # split total feature matrix into train and test sets
        self.trainFeatureMatrix,self.testFeatureMatrix, \
        self.trainTargets,self.testTargets = train_test_split(
            self.totalFeatureMatrix,self.totalTargets,
            test_size=self.config.getTestRatio(),
            shuffle=self.config.isSplitShuffle(),
            random_state=self.config.getShuffleRandomSeed()
        )
        if self.config.isTestWholeSet():
            self.testFeatureMatrix = self.totalFeatureMatrix
            self.testTargets = self.totalTargets

        Log(LOG_INFO) << "Whole size: %d" % self.totalFeatureMatrix.shape[0]
        Log(LOG_INFO) << "Train size: %d" % self.trainFeatureMatrix.shape[0]
        Log(LOG_INFO) << "Test size: %d"  % self.testFeatureMatrix.shape[0]
        return

    def extractVerifyFeatures(self):
        verifyFiles = self.config.getVerifyDataFiles()
        allfm = []
        alltar = []
        for tf in verifyFiles:
            fm,tar = self.getFeatureFromFile(tf)
            allfm.append(fm)
            alltar.append(tar)

        self.verifyFeatureMatrix = np.vstack(allfm)
        self.verifyTargets = np.vstack(alltar)

        if gGeneralConfig.isEnableFeatureStandardization():
            sf = gGeneralConfig.getLoadScalerFile()
            if not os.path.isfile(sf):
                Log(LOG_FATAL) << "You must provide scaler file for verification with standardization."
            self.verifyFeatureMatrix = self.scaler.transform(self.verifyFeatureMatrix)

        return