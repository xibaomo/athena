from modules.mlengine_cores.classifier_cores.svm.svm import SupportVectorMachine
from modules.mlengine_cores.classifier_cores.decisiontree.decisiontree import DecisionTree
from modules.mlengine_cores.classifier_cores.randomforest.randomforest import RandomForest
from modules.mlengine_cores.classifier_cores.dnn.dnn import DNNClassifier
from modules.basics.common.logger import *

EngineCoreSwitcher = {
    0: SupportVectorMachine,
    1: DecisionTree,
    2: RandomForest,
    3: DNNClassifier
    }

def createMLEngineCore(coreType,feature_extractor):
    func = EngineCoreSwitcher[coreType]
    
    if func is None:
        Log(LOG_FATAL) << "Specified classifier core not found: " + str(coreType)
        
    return func(feature_extractor)