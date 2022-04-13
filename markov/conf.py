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

    def getPosProbThreshold(self):
        return self.yamlDict['MARKOV']['POS_PROB_THRESHOLD']

    def getTPReturn(self):
        return self.yamlDict['MARKOV']['TP_RETURN']

    def getSLReturn(self):
        sl = self.yamlDict['MARKOV']['SL_RETURN']
        if sl == 'auto':
            sl = -self.getTPReturn()
        return sl

    def getNumPartitions(self):
        return self.yamlDict['MARKOV']['NUM_PARTITIONS']

    def getOverbuyThd(self):
        return self.yamlDict['MARKOV']['OVERBUY_THD']

    def getOversellThd(self):
        return self.yamlDict['MARKOV']['OVERSELL_THD']