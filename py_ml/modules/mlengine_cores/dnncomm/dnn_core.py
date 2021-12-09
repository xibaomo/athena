'''

fxua

'''
from modules.mlengine_cores.mlengine_core import MLEngineCore
from modules.mlengine_cores.dnncomm.dnnconf import DNNConfig
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.basics.common.logger import *
from modules.mlengine_cores.dnncomm.dnncreator import createDNNModel
import tensorflow as tf
import os
import numpy as np
from sklearn.utils import class_weight

class DNNCore(MLEngineCore):
    def __init__(self,input_dim,est=None):
        super(DNNCore,self).__init__(est)
        self.input_dim = input_dim
        self.config = DNNConfig()
        if not est is None:
            self.estimator = est
        else:
            if input_dim <=0:
                return

            self.createEsimator()
        return

    def _createModel(self):
        Log(LOG_FATAL) << "Should be implemented in concrete class"
        return

    def createEstimator(self):
        self.estimator = self._createModel()
        return

    def loadEstimator(self,est):
        Log(LOG_FATAL) << "Should be implemented in concrete class"
        return

    def train(self,feature_matrix,targets):
        scaler = StandardScaler()
        scaler.fit(feature_matrix)
        self.estimator.layers[0].set_weights([self.estimator.layers[0].get_weights()[0],
                                              -scaler.mean_])
        weights = np.diag(1.0/scaler.scale_)
        self.estimator.layers[1].set_weights([weights,
                                              np.zeros([feature_matrix.shape[1],])])

        
        self.history = self.estimator.fit(feature_matrix,targets,
                                          batch_size = self.config.getBatchSize(),
                                          epochs=self.config.getEpochs(),
                                          shuffle=self.config.isShuffle(),
                                          verbose=self.config.getVerbose(),
                                          class_weight=self.config.getClassWeight()
                                          )

#         Log(LOG_INFO) << "Train with generator ..."
#         fm=feature_matrix
#         tar=targets
#         bs = self.config.getBatchSize()
#         steps = fm.shape[0]*1./bs
#         steps = int(steps) + 1
#         self.estimator.fit_generator(self.loadTrainData(fm, tar, bs),
#                                      steps_per_epoch=steps,
#                                      verbose=self.config.getVerbose(),
#                                      epochs= self.config.getEpochs(),
#                                      shuffle=self.config.isShuffle(),
#                                      use_multiprocessing=True,
#                                      workers=self.config.getNJobs())
        Log(LOG_INFO) << "Final loss: %f" % self.getFinalLoss()
        return

    def predict(self,feature_matrix):
        self.predictedTargets = self.estimator.predict(feature_matrix)
        return

    def getTrainHistory(self):
        return self.history

    def getFinalLoss(self):
        return self.history.history['loss'][-1]
    
    def saveModel(self,mfn):
        super(DNNCore, self).saveDNNModel(mfn,self.estimator)
        return 

    def loadTrainData(self,train_fm,train_tar,batch_size):
        c1 = 0
        c2 = 0
        maxsize = train_fm.shape[0]
        while 1:
            c1 = c2 
            c2 = c1 + batch_size
            if c2 > maxsize:
                c2 = maxsize
            fm = train_fm[c1:c2,:]
            tar= train_tar[c1:c2]
            yield fm,tar
            if c2 == maxsize:
                break
        
