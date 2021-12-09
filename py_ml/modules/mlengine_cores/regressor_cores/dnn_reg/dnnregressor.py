from modules.mlengine_cores.dnncomm.dnncreator import createDNNModel
from modules.mlengine_cores.dnncomm.dnn_core import DNNCore
from modules.basics.common.logger import *
class DNNRegressor(DNNCore):
    def __init__(self,input_dim,est=None):
        super(DNNRegressor,self).__init__(input_dim,est)
        Log(LOG_INFO) << "DNN regressor core created"
        return

    def _createModel(self):
        model = createDNNModel(self.input_dim,None,"mean_squared_error")
        return model

    def createEsimator(self):
        self.estimator = self._createModel()
        return

    def loadEstimator(self,est):
        optm = self.config.getAlgorithm()
        est.compile(loss='mean_squared_error',optimizer=optm,metrics=['accuracy'])
        self.estimator = est
        return