from modules.mlengine_cores.mlengine_core import MLEngineCore
from modules.mlengine_cores.comm.dnnconf import DNNConfig
from modules.basics.common.logger import *
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from modules.basics.conf.mlengineconf import gMLEngineConfig
from keras.wrappers.scikit_learn import KerasClassifier
from modules.mlengine_cores.comm.dnncreator import createDNNModel
class DNNClassifier(MLEngineCore):
    '''
    classdocs
    '''
    def __init__(self, input_dim,est = None):
        super(DNNClassifier, self).__init__(est)
        self.input_dim = input_dim
        if not est is None:
            self.estimator = est
        else:
            self.config =DNNConfig()
            self.createEstimator()
            
            Log(LOG_INFO) <<"DNN classifier is created"
        return

    def _createModel(self):

        model = createDNNModel(self.input_dim, 
                               "sigmoid", "binary_crossentropy")
        
        model.summary()
        return model

    def createEstimator(self):
        self.estimator = KerasClassifier(build_fn = self._createModel,
                                         epochs = self.config.getEpochs(),
                                         batch_size = self.config.getBatchSize(),
                                         verbose = self.config.getVerbose())
        return


    def predict(self,fm):
        super(DNNClassifier,self).predict(fm)
        
        self.predictedTargets = [round(x) for x in self.predictedTargets]
        return

