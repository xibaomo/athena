import yaml

class PortfolioConfig(object):
    def __init__(self, cf):
        self.yamlDict = yaml.load(open(cf), Loader=yaml.FullLoader)
        self.root = "PORTFOLIO"

    def getTargetDate(self):
        dt =  self.yamlDict[self.root]['TARGET_DATE']
        return dt.strftime('%Y-%m-%d')

    def getLookforwardList(self):
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

    def getShortTermWeight(self):
        return self.yamlDict[self.root]['SHORT_TERM_WEIGHT']

    def getVolumeValueType(self):
        return self.yamlDict[self.root]['VOLUME_VALUE_TYPE']

    def getScoreMethod(self):
        return self.yamlDict[self.root]['SCORE_METHOD']
    def getTimeSteps(self):
        return self.yamlDict[self.root]['TIME_STEPS']
    def isRandomSelect(self):
        return self.yamlDict[self.root]['RANDOM_SELECT']
    def getRiskFreeInterestRate(self):
        return self.yamlDict[self.root]['RISK_FREE_RATE']
    def getExcludeList(self):
        try:
            return self.yamlDict[self.root]['EXCLUDE_LIST']
        except:
            return []

class ScoreSettingConfig(object):
    def __init__(self,portconf):
        self.yamlDict = portconf.yamlDict
        self.root = "SCORE_SETTING"
        return

    def getMarkovSpeedLookback(self):
        return self.yamlDict[self.root]['MKV_SPEED']['LOOKBACK']
    def getMarkovSpeedBound(self):
        return self.yamlDict[self.root]['MKV_SPEED']['BOUND']
    def getMarkovSpeedPartitions(self):
        return self.yamlDict[self.root]['MKV_SPEED']['PARTITIONS']
    def getSpecificHeatLookback(self):
        return self.yamlDict[self.root]['SPECIFIC_HEAT']['LOOKBACK']
    def getCDFType(self):
        return self.yamlDict[self.root]['MKV_SPEED']['CDF_TYPE']

class DpminConfig(object):
    def __init__(self,portconf):
        self.yamlDict = portconf.yamlDict
        self.root = "DP_MINIMIZE"

    def getCostType(self):
        return self.yamlDict[self.root]['COST_TYPE']
    def getMaxNumSyms(self):
        return self.yamlDict[self.root]['MAX_NUM_SYMS']
    def getMinNumSyms(self):
        return self.yamlDict[self.root]['MIN_NUM_SYMS']
    def getUBReturn(self):
        ub = self.yamlDict[self.root]['UB_RETURN']
        return float(ub)
    def getLBReturn(self):
        lb = self.yamlDict[self.root]['LB_RETURN']
        return float(lb)
    def getPartitions(self):
        ps =  self.yamlDict[self.root]['PARTITIONS']
        return int(ps)
    def getStationaryCheckDays(self):
        days = self.yamlDict[self.root]['STATIONARY_CHECK_DAYS']
        return int(days)
    def getUpProbWeight(self):
        return self.yamlDict[self.root]['UP_PROB_WEIGHT']
    def getResultRank(self):
        return self.yamlDict[self.root]['RESULT_RANK']
    def getCDFType(self):
        return self.yamlDict[self.root]['CDF_TYPE']