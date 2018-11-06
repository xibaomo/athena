from modules.mlengine_cores.classifier_cores.svm.svm import SupportVectorMachine
from modules.mlengine_cores.classifier_cores.decisiontree.decisiontree import DecisionTree
from modules.mlengine_cores.classifier_cores.randomforest.randomforest import RandomForest
from modules.mlengine_cores.classifier_cores.dnn.dnnclassifier import DNNClassifier
from modules.mlengine_cores.regressor_cores.dnn_reg.dnnregressor import DNNRegressor
from modules.basics.common.logger import *

EngineCoreSwitcher = {
    0: SupportVectorMachine,
    1: DecisionTree,
    2: RandomForest,
    3: DNNClassifier,
    4: DNNRegressor
    }

def createMLEngineCore(coreType,input_dim):
    func = EngineCoreSwitcher.get(coreType,"No such a core type")
    
    if func is None:
        Log(LOG_FATAL) << "Specified classifier core not found: " + str(coreType)
    
    if coreType >= 3:
        return func(input_dim)
    
    return func()