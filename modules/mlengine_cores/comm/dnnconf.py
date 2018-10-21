from modules.basics.conf.topconf import TopConf 
import numpy as np 
class DNNConfig(TopConf): 
    def __init__(self): 
        super(DNNConfig,self).__init__() 
        return 
    def getEpochs(self): 
        return self.yamlDict['EPOCHS'] 
    
    def getNeurons(self): 
        return self.yamlDict['NEURONS']
    
    def getActivation(self): 
        return self.yamlDict['ACTIVATION'] 
    
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
    
    def getlearnRate(self): 
        return self.yamlDict['LEARN_RATE'] 
    
    def getMomentum(self): 
        return self.yamlDict['MOMENTUM'] 
    
    def getVerbose(self):
        return self.yamlDict['VERBOSE']