from modules.basics.conf.topconf import TopConf 
import numpy as np 
class DNNConfig(TopConf): 
    def __init__(self): 
        super(DNNConfig,self)._init_() 
        return 
    def getEpochs(self): 
        return self.yamlDict['EPOCHS'] 
    
    def getNeurons(self): 
        return self.yamlDict['NEURONS']
    
    def getActivation(self): 
        return self.yamlDict['ACTIVATION'] 
    
    def getDropoutRate(self): 
        return self.yamlDict['DROPOUT_RATE'] 
    
    def getBatchSize(self): 
        return self .yamlDict['BATCH_SIZE'] 
    
    def getAlgorithm(self): 
        return self.yamlDict['ALGORITHM'] 
    
    def getWeightinit(self): 
        return self.yamlDict['WEIGHT_INIT'] 
    
    def getlearnRate(self): 
        return self.yamlDict['LEARN_RATE'] 
    
    def getMomentum(self): 
        return self.yamlDict['MOMENTUM'] 