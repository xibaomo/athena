import yaml

class MkvSvmConfig(object):
    def __init__(self, cf):
        self.root = 'MARKOV_SVM'
        self.yamlDict = yaml.load(open(cf), Loader = yaml.FullLoader)

    def getProbCalType(self):
        return self.yamlDict[self.root]['PROB_CAL_TYPE']

    def getLookback(self):
        return self.yamlDict[self.root]['LOOKBACK']

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
    
    def getMinSpeed(self):
        return self.yamlDict[self.root]['MIN_SPEED']
    
    def getFeatureFile(self):
        return self.yamlDict[self.root]['FEATURE_FILE']
    
    def getLabelFile(self):
        return self.yamlDict[self.root]['LABEL_FILE']
    
    def getMinProb(self):
        return self.yamlDict[self.root]['MIN_PROB']
    
    def getMinPosInterval(self):
        return self.yamlDict[self.root]['MIN_POS_INTERVAL']
    
    def getPredictorType(self):
        return self.yamlDict[self.root]['PREDICTOR_TYPE']






