'''
Created on Nov 22, 2018

@author: fxua
'''
from apps.app import App
from modules.basics.common.logger import *
from apps.forex_trainer.fextor.forexfextor import ForexFextor
from apps.forex_trainer.fxtconf import FxtConfig
from modules.mlengine_cores.mlengine_core_creator import createMLEngineCore
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.mlengines.classifier.classifier import Classifier
from modules.basics.conf.generalconf import gGeneralConfig
class ForexFilter(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(ForexFilter,self).__init__()
        self.config = FxtConfig()
        self.fextor = ForexFextor(self.config)
        
        Log(LOG_INFO) << "App: forex filter created"
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):
        self.fextor.loadTickFile()
        self.fextor.computeFeatures()
        fm = self.fextor.getTrainFeatureMatrix()
        input_dim = fm.shape[1]    
        engCore = createMLEngineCore(gMLEngineConfig.getEngineCoreType(), input_dim)   
        self.mlEngine = Classifier(engCore)
        
        Log(LOG_INFO) << "ML engine created"
        return
    
    def execute(self):
        
        fm = self.fextor.getTrainFeatureMatrix()
        tars = self.fextor.getTrainTargets()
        Log(LOG_INFO) << "Training ..."
        self.mlEngine.train(fm, tars)
        Log(LOG_INFO) << "Training done"
        
        Log(LOG_INFO) << "Predicting ..."
        fm = self.fextor.getTestFeatureMatrix()
        self.mlEngine.predict(fm)
        
        self.computeProfit()
        return
    
    def computeProfit(self):
        testSize = len(self.fextor.getTestTargets())
        num_fail = sum(self.fextor.getTestTargets())
        dream_profit = (testSize-num_fail) * self.config.getPointValue() * self.config.getTakeProfit()
        Log(LOG_INFO) << "%d transactions. Dream profit = $%.2f" % (testSize,dream_profit)
        
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
        self.profit = num_good * profitPerTran -  num_miss * lossPerTran
        
        Log(LOG_INFO) << "***************************";
        Log(LOG_INFO) << "********* Summary *********"
        Log(LOG_INFO) << "***************************";
        Log(LOG_INFO) << "Total transactions (original): %d" % (len(true_tar))
        badfrac = sum(true_tar)/float(len(true_tar)) 
        Log(LOG_INFO) << "good: %.1f%%, bad: %.1f%%" % ((1-badfrac)*100,badfrac*100)
        Log(LOG_INFO) << "Actual transactions: %d" % (num_good + num_miss)
        Log(LOG_INFO) << "Profit transactions: %d" % (num_good)
        Log(LOG_INFO) << "Loss transactions: %d" % (num_miss)
        Log(LOG_INFO) << "Total profit: %.2f" % self.profit
        Log(LOG_INFO) << "Dream profit: %.2f" % dream_profit
        Log(LOG_INFO) << "%.2f%% of dream profit taken" % (100*self.profit/dream_profit)
        
        return 
        
    def finish(self):
        od = gGeneralConfig.getOutputDir()
        modelFile = self.config.getModelPrefix() + "_profit_" + str(int(self.profit))
        self.mlEngine.saveModel(od+"/"+modelFile) 
        return   
        
        