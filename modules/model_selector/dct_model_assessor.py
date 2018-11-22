'''
Created on Sep 4, 2018

@author: fxua
'''
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from modules.basics.common.logger import *
from modules.basics.conf.modelselectorconf import gModelSelectorConfig
from modules.mlengine_cores.classifier_cores.decisiontree.decisiontree import DecisionTree
class DCTModelAssessor(object):
    def __init__(self, fm_train, label_train):
        self.feature_matrix = fm_train
        self.labels = label_train
        return

    def prepareParams(self):
        dctConfig = gModelSelectorConfig.getDCTConfig()
        splits = dctConfig.getSplitList()
        params = {
            "criterion": ['gini','entropy'],
            "min_samples_split": splits
            }

        return params

    def assess(self):
        clf = tree.DecisionTreeClassifier()
        params = self.prepareParams()
        grid_search = GridSearchCV(clf, param_grid = params,
                                   n_jobs = gModelSelectorConfig.getNJobs(),
                                   cv = gModelSelectorConfig.getCVFold())
        grid_search.fit(self.feature_matrix, self.labels)

        engCore = DecisionTree(grid_search.best_estimator_)

        return (engCore, grid_search.best_score_)
