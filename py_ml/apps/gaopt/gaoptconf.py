'''
Created on Oct 9, 2018

@author: fxua
'''
from modules.basics.conf.topconf import TopConf
from modules.basics.conf.masterconf import gMasterConfig
class GaOptConfig(TopConf):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        super(GaOptConfig,self).__init__()
        self.loadYamlDict(gMasterConfig.getTotalYamlTree()['GENETIC_OPT'])

        return
    
    def getNumParams(self):
        return self.yamlDict['NUM_PARAMS']
    
    def getUpperBounds(self):
        return self.yamlDict['UPPER_BOUNDS']
    
    def getLowerBounds(self):
        return self.yamlDict['LOWER_BOUNDS']
    
    def getObjectiveType(self):
        return self.yamlDict['OBJECTIVE_TYPE']
    
    def getPopulationSize(self):
        return self.yamlDict['POPULATION_SIZE']
    
    def getNumGenerations(self):
        return self.yamlDict['NUM_GENERATIONS']
    
    def getYamlModifier(self):
        return self.yamlDict['YAML_MODIFIER']
    
    def getYamlTemplate(self):
        return self.yamlDict['YAML_TEMPLATE']
    
    def getSGEJobFile(self):
        return self.yamlDict['SGE_JOB_FILE']
    
    def getCrossProb(self):
        return self.yamlDict['CROSSOVER_PROB']
    
    def getMutateProb(self):
        return self.yamlDict['MUTATE_PROB']
    
    def getIndProb(self):
        return self.yamlDict['INDEPENDENT_PROB']
    
    def getTournamentSize(self):
        return self.yamlDict['TOURNAMENT_SIZE']
    
    def getDataFiles(self):
        return self.yamlDict['DATA_FILES']
    
    def getResultKeywords(self):
        return self.yamlDict['RESULT_KEYWORDS']
    
    def getNItersPerKid(self):
        return self.yamlDict['NITERS_PER_KID']
    
    def getSumBound(self):
        return self.yamlDict['SUM_BOUND']
    
    
    
    
        