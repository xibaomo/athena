'''
Created on Dec 2, 2018

@author: fxua
'''
from apps.app import App
from modules.basics.common.logger import *
from apps.forex_bar_trainer.fbtconf import FbtConfig
from apps.forex_bar_trainer.bar_feature_calculator import BarFeatureCalculator
from modules.mlengine_cores.mlengine_core_creator import createMLEngineCore
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.basics.conf.generalconf import gGeneralConfig
from modules.mlengines.classifier.classifier import Classifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
class ForexBarTrainer(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexBarTrainer,self).__init__()
        self.config = FbtConfig()
        self.fextor = BarFeatureCalculator(self.config)
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
        
    def prepare(self):
        self.fextor.loadMinBars()
        self.fextor.computeFeatures(self.config.getFeatureList())
        self.totalFeatureMatrix,self.totalLabels = self.fextor.getTotalFeatureMatrix()
       
        self.totalLabels = self.getLabel(self.config.getPosType())
        
        testSize = self.config.getTestSize()
        trainSize = len(self.totalLabels) - testSize
        
        self.trainFeatureMatrix = self.totalFeatureMatrix[:trainSize,:]
        self.trainTargets = self.totalLabels[:trainSize]
        self.testFeatureMatrix = self.totalFeatureMatrix[trainSize:,:]
        self.testTargets = self.totalLabels[trainSize:]
        
        Log(LOG_INFO) << "Train size: %d" % trainSize
        Log(LOG_INFO) << "Test size:  %d" % testSize
        input_dim = self.trainFeatureMatrix.shape[1]
        engCore = createMLEngineCore(gMLEngineConfig.getEngineCoreType(),input_dim)
        self.mlEngine = Classifier(engCore)
        return
    
    def execute(self):
        Log(LOG_INFO) << "Training ..."
        self.mlEngine.train(self.trainFeatureMatrix,self.trainTargets)
        Log(LOG_INFO) << "Training done"

        Log(LOG_INFO) << "Predicting ..."
        self.mlEngine.predict(self.testFeatureMatrix)
        Log(LOG_INFO) << "Prediction done"
        
        self.computeProfit()
        return
    
    def getLabel(self,pos_type):
        labels = []
        for t in self.totalLabels:
            if pos_type == "buy":
                if t == 0:
                    labels.append(t)
                else:
                    labels.append(1)
            if pos_type == "sell":
                if t == 1:
                    labels.append(0)
                else:
                    labels.append(1)
        if len(labels) != len(self.totalLabels):
            Log(LOG_FATAL) << "length of labels inconsistent: " + pos_type
            
        return np.array(labels)
            
    def computeProfit(self):
        pred = self.mlEngine.getPredictedTargets()
        profit = 0.
        num_good=0
        num_miss=0
        for i in range(len(pred)):
            if pred[i] == 0 and self.testTargets[i] == 0:
                num_good+=1
                profit += self.config.getProfitLoss()
            if pred[i] == 0 and self.testTargets[i] == 1:
                profit -= self.config.getProfitLoss()
                num_miss+=1
                
        self.profit = profit
        true_tar = self.testTargets
        acc = accuracy_score(true_tar,pred)
        dream_profit = (len(true_tar) - sum(true_tar))*self.config.getProfitLoss()
        Log(LOG_INFO) << "***************************";
        Log(LOG_INFO) << "********* Summary *********"
        Log(LOG_INFO) << "***************************";
        Log(LOG_INFO) << "Total transactions (original): %d" % (len(true_tar))
        badfrac = sum(true_tar)/float(len(true_tar)) 
        Log(LOG_INFO) << "good: %.1f%%, bad: %.1f%%" % ((1-badfrac)*100,badfrac*100)
        Log(LOG_INFO) << "Test set accuracy: %f" % acc
        Log(LOG_INFO) << "Actual transactions: %d" % (num_good + num_miss)
        Log(LOG_INFO) << "Profit transactions: %d" % (num_good)
        Log(LOG_INFO) << "Loss transactions: %d" % (num_miss)
        Log(LOG_INFO) << "Total profit: %.2f" % self.profit
        Log(LOG_INFO) << "Dream profit: %.2f" % dream_profit
        Log(LOG_INFO) << "%.2f%% of dream profit taken" % (100*profit/dream_profit)
        
        return
    
    def finish(self):
        od = gGeneralConfig.getOutputDir()
        modelFile = self.config.getForexSymbol() + "_" + self.config.getPosType() +  \
                    "_profit_" + str(int(self.profit))
        self.mlEngine.saveModel(od+"/"+modelFile) 
        
        self.dumpPrediction()
        return 
    
    def dumpPrediction(self):
        pf = gGeneralConfig.getOutputDir() + "/prediction.csv"
        df = pd.DataFrame()
        df['pred'] = self.mlEngine.getPredictedTargets()
        df['true'] = self.testTargets
        
        df.to_csv(pf,index=False)
        Log(LOG_INFO) << "Prediction dumped to %s" % pf
        
        
        