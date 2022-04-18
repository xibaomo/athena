import yaml

class MarkovConfig(object):
    def __init__(self,cf):
        self.yamlDict = yaml.load(open(cf))

    def getProbCalType(self):
        return self.yamlDict['MARKOV']['PROB_CAL_TYPE']

    def getNumStates(self):
        return self.yamlDict['MARKOV']['NUM_STATES']

    def getLookback(self):
        return self.yamlDict['MARKOV']['LOOKBACK']

    def getLongLookback(self):
        return self.yamlDict['MARKOV']['LONG_LOOKBACK']

    def getPosProbThreshold(self):
        return self.yamlDict['MARKOV']['POS_PROB_THRESHOLD']

    def getUBReturn(self):
        return self.yamlDict['MARKOV']['UB_RETURN']

    def getLBReturn(self):
        sl = self.yamlDict['MARKOV']['LB_RETURN']
        if sl == 'auto':
            sl = -self.getUBReturn()
        return sl

    def getNumPartitions(self):
        return self.yamlDict['MARKOV']['NUM_PARTITIONS']

    def getOverbuyThd(self):
        return self.yamlDict['MARKOV']['OVERBUY_THD']

    def getOversellThd(self):
        return self.yamlDict['MARKOV']['OVERSELL_THD']

    def getSteps(self):
        return self.yamlDict['MARKOV']['STEPS']