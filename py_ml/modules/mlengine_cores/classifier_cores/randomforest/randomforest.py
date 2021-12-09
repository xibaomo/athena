'''
Created on Sep 4, 2018

@author: fxua
'''
from modules.mlengine_cores.mlengine_core import MLEngineCore
from sklearn.ensemble import RandomForestClassifier
from modules.mlengine_cores.classifier_cores.randomforest.rmfconf import RMFConfig
from modules.basics.common.logger import *
from modules.basics.conf.mlengineconf import gMLEngineConfig
from modules.mlengine_cores.sklearn_comm.model_io import saveSklearnModel, \
    loadSklearnModel
from sklearn.utils import class_weight

class RandomForest(MLEngineCore):

    def __init__(self, est = None):
        super(RandomForest, self).__init__(est)

        if est is None:
            self.config = RMFConfig()
            self.config.loadYamlDict(gMLEngineConfig.getYamlDict()['RMF'])
            self.estimator = RandomForestClassifier(n_estimators = self.config.getNEstimator(),
                                                    min_samples_split = self.config.getMinSampleSplit(),
                                                    criterion = self.config.getCriterion(),
                                                    min_samples_leaf = self.config.getMinSamplesLeaf(),
                                                    class_weight=self.config.getClassWeight(),
                                                    n_jobs=self.config.getNJobs(),
                                                    max_depth=self.config.getMaxDepth(),
                                                    random_state=20150513)

            Log(LOG_INFO) << "Classifier: random forest is created: {}".format(self.estimator)
        else:
            self.estimator = est

        return

    def saveModel(self, mfn):
        mfn = mfn + ".pkl"
        saveSklearnModel(mfn, self.estimator)
        Log(LOG_INFO) << "Model dumped: " + mfn
        return

    def loadModel(self, mfn):
        self.estimator = loadSklearnModel(mfn)
        return
    
    def train(self,fm,targets):
        self.estimator.fit(fm,targets)
        print fm.shape
        print self.estimator.feature_importances_
        return
