from modules.mlengine_cores.mlengine_core import MLEngineCore
from modules.mlengine_cores.classifier_cores.dnn.dnnconf import DNNConfig 
from modules.basics.common.logger import * 
from keras.models import Sequential 
from keras.layers import Dense 
import numpy as np 
from modules.basics.conf.mlengineconf import gMLEngineConfig
class DNNClassifier(MLEngineCore): 
    ' ' ' classdocs ' ' ' 
    def __init__(self, est=None,fm_dim1): 
        ' ' ' Constructor ' ' ' 
        super(DNNClassifier,self).__init__(est) 
        if not est is None: 
            self .estimator = est 
        else: 
            self.config = DNNConfig() 
            self.config.loadYamlDict(gMLEngineConfig.getYamlDict()['DNN']) 
            self.estimator = self.createEstimator(fm_dim1) 
            return 
        
        def createEstimator(self,dim1): 
            model = Sequential() 
            neurons = self.config.getNeurons() 
            init_wt = self.config.getWeightinit() 
            act = self .config.getActivation() 
            for n in range(neurons): 
                if n == 0: 
                    model.add(Dense(neurons[n],input_dim=dim1,init=init_wt,activation=act[n]))
                else:
                    model.add(Dense(neurons[n],init=init_wt,activation=act[n]))
                    
            model.add(Dense(1,init=init_wt,activation='sigmoid'))
            optm =self.config.getAlgorithm()
            model.compile(loss='binary_crossentropy',optimizer=optm,metrics=['accuracy'])
            
        def train(self,feature_matrix, targets):
            epochs =self.config.getEpochs()
            bs = self.config.getBatchSize()
            self.estimator.fit(feature_matrix,targets,epochs=epochs,batch_size=bs,verbose=2)
            
            return
        
        def predict(self,feature_matrix):
            y =self .estimator.predict(feature_matrix)
            self.predicted_labels =[round(x) for x in y]
            return
        
        def getPredictedLabelsCself):
            return self.predicted_labels