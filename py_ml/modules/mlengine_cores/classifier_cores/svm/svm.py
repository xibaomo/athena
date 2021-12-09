'''
Created on Sep 4, 2018

@author: fxua
'''
from modules.mlengine_cores.mlengine_core import MLEngineCore
from modules.basics.common.logger import *
from modules.mlengine_cores.classifier_cores.svm.svmconf import SVMConfig
from modules.basics.conf.mlengineconf import gMLEngineConfig
from sklearn import svm

class SupportVectorMachine(MLEngineCore):
    def __init__(self,est=None):
        super(SupportVectorMachine,self).__init__(est)
        if est is None:
            self.svmConfig = SVMConfig()
            self.svmConfig.loadYamlDict(gMLEngineConfig.getYamlDict()['SVM'])
            self.estimator = svm.SVC(kernel=self.svmConfig.getKernel(),
                                     C = self.svmConfig.getRegularParam(),
                                     gamma =  self.svmConfig.getGamma(),
                                     degree = self.svmConfig.getDegree(),
                                     coef0 = self.svmConfig.getCoef0())
            
            Log(LOG_INFO) << "SVM is created: {}".format(self.estimator)
        else:
            self.estimator = est
            
        return
    
