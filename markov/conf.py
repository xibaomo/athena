import yaml

class MarkovConfig(object):
    def __init__(self,cf):
        self.yamlDict = yaml.load(open(cf),Loader=yaml.FullLoader)

    def getProbCalType(self):
        return self.yamlDict['MARKOV']['PROB_CAL_TYPE']

    def getNumStates(self):
        return self.yamlDict['MARKOV']['NUM_STATES']

    def getLookback(self):
        return self.yamlDict['MARKOV']['LOOKBACK']

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

    def getBuyProb(self):
        return self.yamlDict['MARKOV']['BUY_PROB']

    def getSellProb(self):
        return self.yamlDict['MARKOV']['SELL_PROB']

    def getSpeedLookback(self):
        return self.yamlDict['MARKOV']['SPEED_LOOKBACK']

    def getOpenPosSpeed(self):
        return self.yamlDict['MARKOV']['OPEN_POS_SPEED']

    def getPosIntervalMin(self):
        return self.yamlDict['MARKOV']['POS_INTERVAL']

    def getOffPeakSpeedScaler(self):
        return self.yamlDict['MARKOV']['OFF_PEAKSPEED_SCALER']
