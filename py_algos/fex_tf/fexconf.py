import yaml

class FexConfig(object):
    def __init__(self, cf):
        self.root = 'FEX'
        self.yamlDict = yaml.load(open(cf), Loader = yaml.FullLoader)

    def getLookback(self):
        return self.yamlDict[self.root]['LOOKBACK']

    def getLookforward(self):
        return self.yamlDict[self.root]['LOOKFORWARD']

    def getUBReturn(self):
        return self.yamlDict[self.root]['UB_RETURN']

    def getLBReturn(self):
        sl = self.yamlDict[self.root]['LB_RETURN']
        if sl == 'auto':
            sl = -self.getUBReturn()
        return sl

    def getNumPartitions(self):
        return self.yamlDict[self.root]['NUM_PARTITIONS']

    def getPosLifetime(self):
        return self.yamlDict[self.root]['POSITION_LIFETIME']

    def getScalerFile(self):
        return self.yamlDict[self.root]['SCALER_FILE']

    def getModelFile(self):
        return self.yamlDict[self.root]['MODEL_FILE']

    def getFeatureFile(self):
        return self.yamlDict[self.root]['FEATURE_FILE']

    def getLabelFile(self):
        return self.yamlDict[self.root]['LABEL_FILE']

    def getMinPosInterval(self):
        return self.yamlDict[self.root]['MIN_POS_INTERVAL']

    def getPredictorType(self):
        return self.yamlDict[self.root]['PREDICTOR_TYPE']

    def getTrainWeight(self):
        return self.yamlDict[self.root]['TRAIN_WEIGHT']

    def getDataEndPos(self):
        pos = self.yamlDict[self.root]['DATA_END_POS']
        if pos == 0:
            pos = None
        return pos

    def getAllDataSize(self):
        return self.yamlDict[self.root]['ALL_DATA_SIZE']

    def getValidSize(self):
        return self.yamlDict[self.root]['VALID_SIZE']

    def getTestSize(self):
        return self.yamlDict[self.root]['TEST_SIZE']
