from modules.mlengine_cores.mlengine_core import MLEngineCore
from modules.mlengine_cores.classifier_cores.dnn.dnnconf import DNNConfig
from modules.basics.common.logger import *
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from modules.basics.conf.mlengineconf import gMLEngineConfig
from keras.wrappers.scikit_learn import KerasClassifier

class DNNClassifier(MLEngineCore):
    '''
    classdocs
    '''
    def __init__(self, fextor, est = None):
        super(DNNClassifier, self).__init__(fextor,est)
        self.input_dim = input_dim
        if not est is None:
            self.estimator = est
        else:
            self.config =DNNConfig()
            self.config.loadYamlDict(gMLEngineConfig.getYamlDict()['DNN'])
            self.estimator = self.createEstimator()
        return

    def _createModel(self):
        model = Sequential()
        neurons =self.config.getNeurons()
        init_wt =self.config.getWeightlnit()
        act =self.config.getActivation()
        optm =self.config.getAlgorithm()
        model = createDNNModel(self.input_dim, neurons, init_wt, act, "sigmoid", "binary_crossentropy",optm)
        return model

    def createEstimator(self):
        self.estimator =KerasClassifier(build_fn = self._createModel,
                                         epochs = self.config.getEpochs(),
                                         batch_size = self.config.getBatchSize(),
                                         verbose = 0)
        return

    def train(self, feature_matrix, targets):
        self.estimator.fit(feature_matrix, targets)
        return

    def predict(self, feature_matrix):
        y = self.estimator.predict(feature_matrix)
        self.predicted_labels =[round(x) for x in y]
        return

    def getPredictedLabels(self):
        return se1f.predicted_labels
