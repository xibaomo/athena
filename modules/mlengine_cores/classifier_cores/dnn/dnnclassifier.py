from modules.mlengine_cores.mlengine_core import MLEngineCore
from modules.mlengine_cores.dnncomm.dnnconf import DNNConfig
from modules.basics.common.logger import *
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from modules.basics.conf.mlengineconf import gMLEngineConfig
from keras.wrappers.scikit_learn import KerasClassifier
from modules.mlengine_cores.dnncomm.dnncreator import createDNNModel
from modules.mlengine_cores.dnncomm.dnn_core import DNNCore
import pdb
class DNNClassifier(DNNCore):
    '''
    classdocs
    '''
    def __init__(self, input_dim,est = None):
        super(DNNClassifier, self).__init__(est)
        self.input_dim = input_dim
        self.config = DNNConfig()
        if not est is None:
            self.estimator = est
        else:
            if input_dim <= 0:
                return
            super(DNNClassifier,self).createEstimator()
            
            
            Log(LOG_INFO) <<"DNN classifier is created"
        return

    def _createModel(self):

        model = createDNNModel(self.input_dim, 
                               "sigmoid", "binary_crossentropy")
        
        model.summary()
        return model

    def loadEstimator(self,est):
        optm = self.config.getAlgorithm()
        est.compile(loss='binary_crossentropy',optimizer=optm,metrics=['accuracy'])
        self.estimator = est
        return

    def predict(self,fm):
#         pdb.set_trace()
        super(DNNClassifier,self).predict(fm)
        
        self.predictedTargets = [int(round(x)) for x in self.predictedTargets]
        return

