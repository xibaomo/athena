from modules.basics.common.logger import *
from modules.basics.conf.topconf import TopConf
import os

class GeneralConfig(TopConf):
    def __init__(self):
        super(GeneralConfig,self).__init__()
        self.localConfigFile = None
        self.modelTag = None
        return

    def loadCmdLineOpts(self,args):
        self.localConfigFile = args.pos_arg
        self.setModelTag(args.model_tag)
        return

    def setModelTag(self,tag):
        self.modelTag = tag
        return

    def getLocalConfigFile(self):
        return self.localConfigFile

    def getModelTag(self):
        return self.modelTag

    def getLogLevel(self):
        return self.yamlDict['LOG_LEVEL']
    
    def getAppType(self):
        return self.yamlDict['APPLICATION']
    
    def getGaugeFileFormat(self):
        return self.yamlDict['GAUGE_FILE_FORMAT']
    
    def isEnableModelSelector(self):
        return self.yamlDict['ENABLE_MODEL_SELECTOR']
    
    def getMaxNumModels(self):
        return self.yamlDict['MAX_NUM_MODELS']
    
    def getMinNumModels(self):
        return self.yamlDict['MIN_NUM_MODELS']
    
    def getSaveModelFile(self):
        return self.yamlDict['SAVE_MDOEL_FILE']
    
    def getLoadModelFile(self):
        return self.yamlDict['LOAD_MODEL_FILE'] 

    def isEnableTFGraph(self):
        return self.yamlDict['ENABLE_TF_GRAPH']

    def isVerification(self):
        return self.yamlDict['ENABLE_VERIFICATION']

    def getSaveScalerFile(self):
        return self.yamlDict['SAVE_SCALER_FILE']

    def getLoadScalerFile(self):
        return self.yamlDict['LOAD_SCALER_FILE']

    def isEnablePCA(self):
        return self.yamlDict['ENABLE_PCA']

    def getPCAComponents(self):
        return self.yamlDict['PCA_COMPONENTS']

    def getSavePCAFile(self):
        return self.yamlDict['SAVE_PCA_FILE']

    def getLoadPCAFile(self):
        return self.yamlDict['LOAD_PCA_FILE']

    def isEnableFeatureStandardization(self):
        return self.yamlDict['ENABLE_FEATURE_STANDARDIZATION']
    
    def getOutputDir(self):
        fd = self.yamlDict['OUTPUT_DIR']
        if not os.path.isdir(fd):
            os.mkdir(fd)
            
        return fd           

    def integrityCheck(self):
        if self.isVerification() and self.getLoadModelFile() == "":
            Log(LOG_FATAL) << "Verification is enabled. Model file must be provided"
        return
global gGeneralConfig
gGeneralConfig = GeneralConfig()