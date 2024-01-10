import yaml

class PortfolioConfig(object):
    def __init__(self, cf):
        self.yamlDict = yaml.load(open(cf), Loader=yaml.FullLoader)
        self.root = "PORTFOLIO"

    def getTargetDate(self):
        dt =  self.yamlDict[self.root]['TARGET_DATE']
        return dt.strftime('%Y-%m-%d')

    def getLookforward(self):
        return self.yamlDict[self.root]['LOOKFORWARD']

    def getLookback(self):
        return self.yamlDict[self.root]['LOOKBACK']
    def getCapitalAmount(self):
        return self.yamlDict[self.root]['CAPITAL_AMOUNT']

    def getNumSymbols(self):
        return self.yamlDict[self.root]['NUM_SYMBOLS']

    def getSymFile(self):
        return self.yamlDict[self.root]['SYMBOL_FILE']
    def getMAWindow(self):
        return self.yamlDict[self.root]['MA_WINDOW']
    def getWeightBound(self):
        return self.yamlDict[self.root]['WEIGHT_BOUND']

    def getSymbols(self):
        return self.yamlDict[self.root]['SYMBOLS']

    def getSymWeights(self):
        return self.yamlDict[self.root]['WEIGHTS']

    def getMuWeight(self):
        return self.yamlDict[self.root]['MU_WEIGHT']

    def getCostType(self):
        return self.yamlDict[self.root]['COST_TYPE']

    def getSigmaBounds(self):
        return self.yamlDict[self.root]['SIGMA_BOUNDS']
