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
    def getOpenPositionReturn(self):
        return self.yamlDict[self.root]['OPEN_POSITION_RETURN']
    def getClosePositionReturn(self):
        return self.yamlDict[self.root]['CLOSE_POSITION_RETURN']
