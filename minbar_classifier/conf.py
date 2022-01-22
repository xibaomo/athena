import yaml

class MasterConf(object):
    def __init__(self,cf):
        self.yamlDict = yaml.load(open(cf))

    def setRoot(self,root_key):
        self.yamlDict = self.yamlDict[root_key]

    def getPosLifeSec(self):
        return self.yamlDict['LABELING']['POSITION_LIFETIME']*3600*24

    def getReturnThreshold(self):
        return self.yamlDict['LABELING']['RETURN_THRESHOLD']

    def getTrueReturnRatio(self):
        return self.yamlDict['LABELING']['TRUE_RETURN_RATIO']

    def getTestSize(self):
        return self.yamlDict['TRAINING']['TEST_SIZE']

    def getScalerFile(self):
        return self.yamlDict['TRAINING']['SCALER_FILE']

    def getTestStartDate(self):
        return self.yamlDict['TRAINING']['TEST_START_DATE']

    def getTestEndDate(self):
        return self.yamlDict['TRAINING']['TEST_END_DATE']

    def getModelType(self):
        return self.yamlDict['MODEL']['TYPE']

    def getMLModelFile(self):
        return self.yamlDict['MODEL']['ML_MODEL_FILE']
    def getTFModelFile(self):
        return self.yamlDict['MODEL']['TF_MODEL_FILE']

    def getDNNEpochs(self):
        return self.yamlDict['MODEL']['DNN']['EPOCHS']

    def getFeatureType(self):
        return self.yamlDict['FEATURES']['TYPE']
