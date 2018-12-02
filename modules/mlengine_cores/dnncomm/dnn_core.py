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

        cp_callback=None
        if self.config.getCheckPointPeriod() > 0:
            if not os.path.isdir(self.config.getCheckPointFolder()):
                os.mkdir(self.config.getCheckPointFolder())

            ck_path = self.config.getCheckPointFolder() + "/cp-{epoch:04d}.ckpt"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                ck_path,verbose=1,save_weights_only=True,
                period=self.config.getCheckPointPeriod()
            )
        self.history = self.estimator.fit(feature_matrix,targets,
                                          batch_size = self.config.getBatchSize(),
                                          epochs=self.config.getEpochs(),
                                          shuffle=self.config.isShuffle(),
                                          verbose=self.config.getVerbose(),
                                          class_weight={0:1,1:1}
                                          )

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

