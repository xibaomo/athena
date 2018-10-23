'''
Created on Oct 22, 2018

@author: fxua
'''
from apps.app import App
from apps.generic_app.gaconf import GaConfig
from apps.generic_app.gafextor import GaFextor
from modules.mlengine_cores.classifier_cores.dnn.dnn import DNNClassifier
from modules.mlengines.classifier.classifier import Classifier
from modules.basics.common.logger import *
class GenericApp(App):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(GenericApp,self).__init__()
        self.config = GaConfig()
        self.fextor = GaFextor(self.config)
        return
    
    @classmethod
    def getInstance(cls):
        return cls()
    
    def prepare(self):
        self.fextor.prepare()
        fm = self.fextor.getTrainFeatureMatrix()
        input_dim = fm.shape[1]
        
        engCore = DNNClassifier(input_dim)
        self.engine = Classifier(engCore)
        return
    
    def execute(self):
        Log(LOG_INFO) << "Training ..."
        fm = self.fextor.getTrainFeatureMatrix()
        labels = self.fextor.getTrainTargets()
        self.engine.train(fm, labels)
        
        Log(LOG_INFO) << "Predicting ..."
        fm = self.fextor.getTestFeatureMatrix()
        true_labels = self.fextor.getTestTargets()
        self.engine.predict(fm)
        self.engine.evaluatePrediction(fm, true_labels)
        return
    
    def finish(self):
        return
    