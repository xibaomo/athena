import yaml

class GraphloopConfig(object):
    def __init__(self, cf):
        self.yamlDict = yaml.load(open(cf), Loader=yaml.FullLoader)
        self.root = "GRAPHLOOP"

    def getEndNode(self):
        return self.yamlDict[self.root]['END_NODE']

    def getForexListFile(self):
        return self.yamlDict[self.root]['FOREX_LIST_FILE']
    def getSelectedNodes(self):
        return self.yamlDict[self.root]['SELECTED_NODES']
    def getBuyThresholdReturn(self):
        return self.yamlDict[self.root]['BUY_THRESHOLD_RETURN']
    def getSellThresholdReturn(self):
        return self.yamlDict[self.root]['SELL_THRESHOLD_RETURN']

    def isAllowPositions(self):
        return self.yamlDict[self.root]['ALLOW_POSITIONS']
