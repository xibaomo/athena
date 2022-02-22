import yaml

class MarkovConfig(object):
    def __init__(self,cf):
        self.yamlDict = yaml.load(open(cf))

    # def getReturnBounds(self):
    #     return self.yamlDict['MARKOV']['RETURN_BOUNDS']
    #
    # def getOptAlgo(self):
    #     return self.yamlDict['MARKOV']['OPTIMIZATION']

    # def getOptTarget(self):
    #     return self.yamlDict['MARKOV']['OPT_TARGET']
    def getProbCalType(self):
        return self.yamlDict['MARKOV']['PROB_CAL_TYPE']

    def getNumStates(self):
        return self.yamlDict['MARKOV']['NUM_STATES']

    def getLookback(self):
        return self.yamlDict['MARKOV']['LOOKBACK']

    def getPosProbThreshold(self):
        return self.yamlDict['MARKOV']['POS_PROB_THRESHOLD']

    def getTPReturn(self):
        return self.yamlDict['MARKOV']['TP_RETURN']

    def getSLReturn(self):
        sl = self.yamlDict['MARKOV']['SL_RETURN']
        if sl == 'auto':
            sl = -self.getTPReturn()
        return sl