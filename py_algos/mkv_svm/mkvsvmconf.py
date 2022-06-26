import yaml

class MkvSvmConfig(object):
    def __init__(self, cf):
        self.yamlDict = yaml.load(open(cf), Loader = yaml.FullLoader)

    def getProbCalType(self):
        return self.yamlDict['MARKOV_SVM']['PROB_CAL_TYPE']

    def getLookback(self):
        return self.yamlDict['MARKOV_SVM']['LOOKBACK']

    def getUBReturn(self):
        return self.yamlDict['MARKOV_SVM']['UB_RETURN']

    def getLBReturn(self):
        sl = self.yamlDict['MARKOV_SVM']['LB_RETURN']
        if sl == 'auto':
            sl = -self.getUBReturn()
        return sl

    def getNumPartitions(self):
        return self.yamlDict['MARKOV_SVM']['NUM_PARTITIONS']

    def getPosLifetime(self):
        return self.yamlDict['MARKOV_SVM']['POSITION_LIFETIME']
    
    def getScalerFile(self):
        return self.yamlDict['MARKOV_SVM']['SCALER_FILE']
    
    def getModelFile(self):
        return self.yamlDict['MARKOV_SVM']['MODEL_FILE']
    
    def getMinSpeed(self):
        return self.yamlDict['MARKOV_SVM']['MIN_SPEED']
    
    def getFeatureFile(self):
        return self.yamlDict['MARKOV_SVM']['FEATURE_FILE']
    
    def getLabelFile(self):
        return self.yamlDict['MARKOV_SVM']['LABEL_FILE']
    
    def getSvm_C(self):
        return self.yamlDict['MARKOV_SVM']['SVM_C']






