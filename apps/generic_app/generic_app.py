'''
Created on Oct 22, 2018

@author: fxua
'''
from apps.app import App
from apps.generic_app.gaconf import GaConfig
from apps.generic_app.gafextor import GaFextor
from apps.generic_app.engine_creator import createEngine,loadEngine
from modules.basics.conf.generalconf import gGeneralConfig
from modules.mlengine_cores.classifier_cores.dnn.dnnclassifier import DNNClassifier
from modules.mlengines.classifier.classifier import Classifier
from modules.basics.common.logger import *
from modules.basics.conf.mlengineconf import gMLEngineConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from scipy import stats
import pdb

class GenericApp(App):
    def __init__(self):
        '''
        Constructor
        '''
        super(GenericApp,self).__init__()
        self.config = GaConfig()
        self.featureExtractor = GaFextor(self.config)
        return
    
    @classmethod
    def getInstance(cls):
        return cls()


    def prepare(self):
        if not gGeneralConfig.isVerification():
            self.featureExtractor.extractTrainFeatures()
            Log(LOG_INFO) << "Features of train set extracted"
        else:
            self.featureExtractor.extractVerifyFeatures()
            Log(LOG_INFO) << "Features of verify set extracted"

        if gGeneralConfig.getLoadModelFile() == "":
            if gGeneralConfig.isVerification():
                Log(LOG_FATAL) << "Verification needs model file"

            fm = self.featureExtractor.getTrainFeatureMatrix()
            self.engine = createEngine(self.config.getEngineType(),
                                       gMLEngineConfig.getEngineCoreType(),
                                       fm.shape[1])
        else:
            self.engine,_,_ = loadEngine(self.config.getEngineType(),
                                         gMLEngineConfig.getEngineCoreType(),
                                         gGeneralConfig.getLoadModelFile())
            Log(LOG_INFO) << "Model loaded from " + gGeneralConfig.getLoadModelFile()
        return

    def execute(self):
        if not gGeneralConfig.isVerification():
            train_fm = self.featureExtractor.getTrainFeatureMatrix()
            train_tg = self.featureExtractor.getTrainTargets()

            if np.isnan(np.min(train_fm)):
                Log(LOG_FATAL) << "Nan is found in train feature matrix"
            if np.isnan(np.min(train_tg)):
                Log(LOG_FATAL) << "Nan is found in train targets"

            Log(LOG_INFO) << "Training ..."
            self.engine.train(train_fm,train_tg)
            Log(LOG_INFO) << "Training done"

            test_fm = self.featureExtractor.getTestFeatureMatrix()
            test_tg = self.featureExtractor.getTestTargets()
        else:
            test_fm = self.featureExtractor.getVerifyFeatureMatrix()
            test_tg = self.featureExtractor.getVerifyTargets()

        Log(LOG_INFO) << "Predicting ..."
        self.engine.predict(test_fm)
        pred_tg = self.engine.getPredictedTargets()
        Log(LOG_INFO) << "prediction done"

        if test_tg is None:
            return

        self.estimateFitErr(test_tg,pred_tg)

        fm_headers = self.config.getFeatureHeaders()
        fm_headers = fm_headers.split(',')

        if not self.config.getResultFileName() == "":
            utils.dumpFeatureTargets(self.config.getResultFileName(),
                                     test_fm,pred_tg,test_tg,
                                     fm_headers,self.config.getTargetHeader())
            Log(LOG_INFO) << "Predicted targets & features dumped to: " + self.config.getResultFileName()
        return

    def estimateFitErr(self,test_tg,pred_tg):
        pred_tg = pred_tg.reshape(-1,1)
        err = test_tg - pred_tg
        rms = utils.computeRMS(err)
        rng = utils.computeRange(err)
        Log(LOG_INFO) << "Error rms on test set: %.2f" % rms
        Log(LOG_INFO) << "Error range on test set: %.2f" % rng

        Log(LOG_INFO) << "Mean of Error: %.4f, std: %.4f" % (np.mean(err),np.std(err))

        _,pv = stats.normaltest(err)

        Log(LOG_INFO) << "p-value of normal dist: %f" % pv
        return

    def finish(self):
        return
    
    def evaluteModel(self):
        x = self.fextor.getTotalFeatureMatrix()
        y = self.fextor.getTotalTargets()
        
        estimators = []
        est = self.core.getEstimator()
        estimators.append(("mlp",est))
        pipeline = Pipeline(estimators)
        seed = 7
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
        results=cross_val_score(pipeline,x,y,cv=kfold)
          
        Log(LOG_INFO) <<"Model eval: %.2f%% (%.2f%%)" % (results.mean()*100.,results.std()*100.)

        
        
        