from modules.basics.conf.topconf import TopConf 
from modules.basics.conf.masterconf import gMasterConfig
import numpy as np 
import ast
class DNNConfig(TopConf): 
    def __init__(self): 
        super(DNNConfig,self).__init__() 
        self.loadYamlDict(gMasterConfig.getTotalYamlTree()['ML_ENGINE_CORE']['DNN'])
        return 
    def getEpochs(self): 
        return self.yamlDict['EPOCHS'] 
    
    def getNeurons(self): 
        return self.yamlDict['NEURONS']
    
    def getActivation(self): 
        return self.yamlDict['ACTIVATION'] 
    
    def getOutputAct(self):
        return self.yamlDict['OUTPUT_ACT']
    
    def getDropoutRate(self): 
        return self.yamlDict['DROPOUT_RATE'] 
    
    def getRegularizer(self):
        return self.yamlDict['REGULARIZER']
    
    def getBatchSize(self): 
        return self.yamlDict['BATCH_SIZE'] 
    
    def getAlgorithm(self): 
        return self.yamlDict['ALGORITHM'] 
    
    def getWeightInit(self): 
        return self.yamlDict['WEIGHT_INIT'] 
    
    def getLearnRate(self): 
        return self.yamlDict['LEARN_RATE'] 
    
    def getLossFunc(self):
        return self.yamlDict['LOSS_FUNC']

    def isShuffle(self):
        return self.yamlDict['ENABLE_SHUFFLE']
    def getMomentum(self): 
        return self.yamlDict['MOMENTUM'] 
    
    def getVerbose(self):
        return self.yamlDict['VERBOSE']

    def getCheckPointPeriod(self):
        return self.yamlDict['CHECKPOINT_PERIOD']

    def getCheckPointFolder(self):
        return self.yamlDict['CHECKPOINT_FOLDER']
    
    def getClassWeight(self):
        val= self.yamlDict['CLASS_WEIGHT']
        if val[0] == '{':
            val = ast.literal_eval(val)
        
        return val
    
    def getNJobs(self):
        return self.yamlDict['N_JOBS']