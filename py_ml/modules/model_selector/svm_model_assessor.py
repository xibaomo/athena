'''
Created on Sep 3, 2018

@author: fxua
'''
from sklearn.model_selection import cross_val_score
from scipy.optimize import minimize
from modules.basics.common.logger import *
from sklearn.model_selection import GridSearchCV
import numpy as np
from modules.basics.conf.modelselectorconf import gModelSelectorConfig
from modules.mlengine_cores.classifier_cores.svm.svm import SupportVectorMachine

class SVMModelAssessor(object):
    '''
    This class is used to assess SVM models.
    For a given kernel type, run grid search to find out the optimal model
    '''

    def __init__(self, fm_train, label_train):
        self.feature_matrix = fm_train
        self.labels = label_train
        return

    def prepareParams(self, kernel):
        g0 = 1./self.feature_matrix.shape[1]
        glist = np.linspace(0.01*g0, 10*g0, 10)
        clist = np.linspace(1, 2000, 30)
        basicParams = {
            "kernel": [kernel],
            "C": clist,
            "gamma": glist
            }

        if kernel == 'rbf' or kernel == 'linear':
            return basicParams
        if kernel == 'poly':
            basicParams['degree'] = [3]
            basicParams['coef0'] = [0.,0.5,1.0]
            return basicParams
        if kernel == 'sigmoid':
            basicParams['coef0']=[0.,0.5,1.0]
            return basicParams

    def assess(self, kernel):
        clf = svm.SVC()
        params = self.prepareParams(kernel)
        grid_search = GridSearchCV(clf, param_grid = params, n_jobs = gModelSelectorConfig.getNJobs(),
                                   cv = gModelSelectorConfig.getCVFold())
        grid_search.fit(self.feature_matrix, self.labels)

        engCore = SupportVectorMachine(grid_search.best_estimator_)

        return (engCore, grid_search.best_score_)

