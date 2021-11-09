import yaml

class MasterConf(object):
    def __init__(self,cf):
        self.yamlDict = yaml.load(open(cf))
    def getPosLifeSec(self):
        return self.yamlDict['LABELING']['POSITION_LIFETIME']*3600*24

    def getReturnThreshold(self):
        return self.yamlDict['LABELING']['RETURN_THRESHOLD']

    def getTestSize(self):
        return self.yamlDict['TRAINING']['TEST_SIZE']

    def getScalerFile(self):
        return self.yamlDict['TRAINING']['SCALER_FILE']

    def getModelFile(self):
        return self.yamlDict['TRAINING']['MODEL_FILE']
