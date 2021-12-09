import pickle

def loadSklearnModel(modelFileName):
    loadedModel = pickle.load(open(modelFileName,'rb'))
    return loadedModel

def saveSklearnModel(modelFileName,model):
    pickle.dump(model,open(modelFileName,'wb'))
    return