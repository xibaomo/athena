'''
Created on Sep 5, 2018

@author: fxua
'''

from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from modules.basics.conf.modelselectorconf import gModelSelectorConfig
from modules.mlengine_cores.classifier_cores.randomforest.randomforest import RandomForest

class RMFModelAssessor(object):
    '''
    classdocs
    '''


    def __init__(self, fm_train,label_train):
        '''
        Constructor
        '''
        self.feature_matrix = fm_train
        self.labels = label_train
        return
    
    def prepareParams(self):
        rmfConfig = gModelSelectorConfig.getRMFConfig()
        n_est = rmfConfig.getNEstimators()
        splits = rmfConfig.getMinSamplesSplit()
        crit = rmfConfig.getCriterion()
        params = {
            'n_estimators': n_est,
            'criterion': crit,
            'min_samples_split': splits
            }
        
        return params
    
    def assess(self):
        clf = RandomForestClassifier()
        params = self.prepareParams()
        grid_search = GridSearchCV(clf,param_grid=params,
                                   n_jobs=gModelSelectorConfig.getNJobs(),
                                   cv = gModelSelectorConfig.getCVFold())
        grid_search.fit(self.feature_matrix,self.labels)
        
        engCore = RandomForest(grid_search.best_estimator_)
        
        return (engCore,grid_search.best_score_)
        
        