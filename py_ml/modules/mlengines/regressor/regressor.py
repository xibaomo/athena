from modules.mlengines.mlengine import MLEngine
from modules.basics.common.logger import *

class Regressor(MLEngine):

    def __init__(self,engineCore=None):
        super(Regressor,self).__init__(engineCore)
        Log(LOG_INFO) << "Regressor is created"
        return

    def predict(self,fm):
        self.engineCore.predict(fm)
        return