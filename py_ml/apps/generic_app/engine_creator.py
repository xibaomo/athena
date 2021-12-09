'''

fxua

'''
from modules.mlengines.regressor.regressor import Regressor
from modules.mlengines.classifier.classifier import Classifier
from modules.mlengine_cores.mlengine_core_creator import  createMLEngineCore
from modules.mlengine_cores.mlengine_core import  MLEngineCore
from modules.mlengine_cores.dnncomm.dnncreator import loadDNNModel
from modules.mlengine_cores.sklearn_comm.model_io import loadSklearnModel

EngineSwitcher = {
    0: Regressor,
    1: Classifier
}
def createEngine(engineType, coreType, input_dim=0):
    if engineType == 0 and coreType == 3:
        coreType = 4

    core = createMLEngineCore(coreType, input_dim)
    func = EngineSwitcher.get(engineType, "No such an engine")

    return func(core)

def loadEngine(engineType,coreType,modelFileName):
    core = createMLEngineCore(coreType,0)
    est = None

    if coreType <=2:
        print "load non-dnn model"
        est = loadSklearnModel(modelFileName)
    else:
        est = loadDNNModel(modelFileName)
    core.loadEstimator(est)
    func = EngineSwitcher.get(engineType,"No such an engine")

    return (func(core),core,est)
