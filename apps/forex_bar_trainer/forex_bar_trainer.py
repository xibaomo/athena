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
from apps.forex_bar_trainer.bar_feature_calculator import GLOBAL_PROB_PERIOD
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection._split import KFold
from sklearn.model_selection._validation import cross_val_score
from scipy.stats import binom
from sklearn.preprocessing import StandardScaler
import owls
import pdb
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
        self.fextor.setLookback(self.config.getLookBack())
        self.fextor.setMALookback(self.config.getMALookBack())
        self.fextor.loadMinBars(self.config.getBarFile())
        self.fextor.computeFeatures(self.config.getFeatureList())
        self.totalFeatureMatrix,self.totalLabels = self.fextor.getTotalFeatureMatrix()
       
#         scaler = StandardScaler()
#         self.totalFeatureMatrix = scaler.fit_transform(self.totalFeatureMatrix)
        
        self.totalLabels = self.getLabel(self.config.getPosType())
        
        testSize = self.config.getTestSize()
        trainSize = len(self.totalLabels) - testSize
        
        self.trainFeatureMatrix = self.totalFeatureMatrix[:trainSize,:]
        self.trainTargets = self.totalLabels[:trainSize]
        self.testFeatureMatrix = self.totalFeatureMatrix[trainSize:,:]
        self.testTargets = self.totalLabels[trainSize:]
#         self.testTime = self.fextor.getTime()[trainSize:]
        
#         pdb.set_trace()
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

        pred = None
#         self.evalModel()
        
        Log(LOG_INFO) << "Predicting ..."
        
        if "BINOM" in self.config.getFeatureList():
            pred = self.predict_binom()
        else:
            self.mlEngine.predict(self.testFeatureMatrix)
            
        Log(LOG_INFO) << "Prediction done"
        
        self.computeProfit(pred)
        self.pred = pred
        return
    
    def predict_binom(self):
#         pdb.set_trace()
        testSize =  self.testFeatureMatrix.shape[0]
        regLen = self.config.getRegFeatureSize()
        Log(LOG_INFO) << "Reg feature size: %d" % regLen

        lookback = self.config.getLookBack()
        labels = self.trainTargets
        p = self.fextor.getBinomProb()
        pred = []
        for i in range(testSize):
            Log(LOG_DEBUG) << "predicting i = %d" % i 
            f = self.testFeatureMatrix[i,:regLen]
            tr,ts = owls.compLastBinom(labels.astype(float),lookback,GLOBAL_PROB_PERIOD)
            bm = np.array([tr,ts])
            f = np.append(f,bm)
            
            Log(LOG_DEBUG) << "binom done, start to predict .."
            if "BMFFT" in self.config.getFeatureList():
                lb = labels[-lookback:]
                ff = self.fextor.compLabelFFT(lb)
                f = np.append(f,ff)
            
            self.mlEngine.predict(f.reshape(1,-1))
            
            Log(LOG_DEBUG) << "prediction done"
            new_label = self.mlEngine.getPredictedTargets()[0]
            labels = np.append(labels, new_label)
            pred.append(new_label)
            
            Log(LOG_DEBUG) << "new label appended. total labels: %d" % len(labels)
            
        
        return np.array(pred)
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
            
    def computeProfit(self,pred=None):
        if pred is None:
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
        Log(LOG_INFO) << "buy: %.1f%%, sell: %.1f%%" % ((1-badfrac)*100,badfrac*100)
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
        
        self.dumpPrediction(self.pred)
        return 
    
    def dumpPrediction(self,pred=None):
        pf = gGeneralConfig.getOutputDir() + "/prediction.csv"
        df = pd.DataFrame()
        
        testSize = self.config.getTestSize()
        alltime = self.fextor.getTime()
        self.testTime = alltime[-testSize:]
        df['time'] = self.testTime
        
        df['true'] = self.testTargets
        
        if pred is None:
            df['pred'] = self.mlEngine.getPredictedTargets()
        else:
            df['pred'] = pred
        
        df.to_csv(pf,index=False)
        Log(LOG_INFO) << "Prediction dumped to %s" % pf
        
    def evalModel(self):
        Log(LOG_INFO) << "Evaluate CV score ..."
        
        kfold = KFold(n_splits=10,shuffle=False)
        res = cross_val_score(self.mlEngine.getEstimator(),self.totalFeatureMatrix,
                              self.totalLabels,cv=kfold,n_jobs=-1)
        Log(LOG_INFO) << "CV accuracy: %f" % res.mean()
        
        
        
        
        
        
        
        
        
        