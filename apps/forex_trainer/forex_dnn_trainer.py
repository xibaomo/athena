'''
Created on Oct 21, 2018

@author: fxua
'''

from apps.app import App
from apps.forex_trainer.fextor.forexfextor import ForexFextor
from apps.forex_trainer.fxtconf import FxtConfig
from modules.basics.common.logger import *
from modules.mlengine_cores.classifier_cores.dnn.dnnclassifier import DNNClassifier
from modules.mlengines.classifier.classifier import Classifier

class ForexDNNTrainer(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexDNNTrainer,self).__init__()
        self.config = FxtConfig()
        self.fextor = ForexFextor(self.config)
        
        Log(LOG_INFO) << "Forex DNN trainer created"
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):
        self.fextor.loadTickFile()
        self.fextor.computeFeatures()
        
        # create classifier
        input_dim = self.fextor.getTrainFeatureMatrix().shape[1]
        engCore = DNNClassifier(input_dim)
        self.mlEngine = Classifier(engCore)
        return
    
    def execute(self):
        Log(LOG_INFO) << "Training ..."
        fm = self.fextor.getTrainFeatureMatrix()
        tar = self.fextor.getTrainTargets()
        self.mlEngine.train(fm, tar)
        
        Log(LOG_INFO) << "Predicting ..."
        fm = self.fextor.getTestFeatureMatrix()
        true_tar = self.fextor.getTestTargets()
        
        self.mlEngine.predict(fm)
        self.mlEngine.evaluatePrediction(fm, true_tar)
        
        self.computeProfit()
        
        return
    
    def finish(self):
        return
    
    def computeProfit(self):
        num_good=0
        num_miss=0
        true_tar = self.fextor.getTestTargets()
        pred_tar  = self.mlEngine.getPredictedTargets()
        if len(true_tar) != len(pred_tar):
            Log(LOG_FATAL) << "Inconsistent size between true and predicted targets"
            
        for i in range(len(pred_tar)):
            if pred_tar[i] == 0:
                if true_tar[i] == 0:
                    num_good+=1
                if true_tar[i] == 1:
                    num_miss+=1
        
        profitPerTran = self.config.getPointValue() * self.config.getTakeProfit()
        lossPerTran = self.config.getPointValue() * self.config.getStopLoss()
        num_fail = sum(true_tar)
        dream_profit = profitPerTran * (len(true_tar) - num_fail)
        profit = num_good * profitPerTran -  num_miss * lossPerTran
        
        Log(LOG_INFO) << "Profit transactions: %d" % num_good
        Log(LOG_INFO) << "Loss transactions: %d" % num_miss
        Log(LOG_INFO) << "Total profit: %.2f" % profit
        Log(LOG_INFO) << "Dream profit: %.2f" % dream_profit
        Log(LOG_INFO) << "%.2f%% of dream profit taken" % (100*profit/dream_profit)
        
        return profit