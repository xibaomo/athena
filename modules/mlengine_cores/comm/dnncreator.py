'''
Created on Sep 8, 2018

@author: fxua
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.models import model_from_yaml
from modules.basics.conf.generalconf import gGeneralConfig
import os
from modules.basics.common.logger import *

Regularizer_switcher = {
    'L1': regularizers.l1,
    'L2': regularizers.l2,
    'L1L2': regularizers.l1_l2
    }

def createDNNModel(input_dim, neurons, init_wt, dropouts,regs, 
                   act, end_act, loss, optm_algo,outnodes=1):
    model = Sequential()
    for n in range(len(neurons)):
        reg,rv=regs[n].split(":")
        regler = Regularizer_switcher[reg]
        if n == 0:
            model.add(Dense(neurons[n], input_dim = input_dim, 
                            kernel_initializer=init_wt, 
                            kernel_regularizer=regler(float(rv)),
                            activation = act[n]))
        else:
            model.add(Dense(neurons[n], kernel_initializer=init_wt,
                            kernel_regularizer=regler(float(rv)), 
                            activation = act[n]))
        
        model.add(Dropout(dropouts[n]))
    model.add(Dense(outnodes, kernel_initializer=init_wt, activation = end_act))
    model.compile(loss = loss, optimizer = optm_algo, metrics=['accuracy'])

    return model

def saveDNNModel(model,modelfile):
    mf,ext = os.path.splitext(modelfile)
    if ext == ".yaml" or ext == ".yml":
        model_yaml = model.to_yaml()
        with open(modelfile,'w') as f:
            f.write(model_yaml)
        model.save_weights(mf+".h5")
        
        Log(LOG_INFO) << "Model saved to %s.yaml & .h5" % mf
        
    else:
        Log(LOG_FATAL) << "%s not supported" % ext
        
    return

def loadDNNModel(modelfile):
    mf,ext = os.path.splitext(modelfile)
    if ext == ".yaml" or ext == ".yml":
        with open(modelfile,'r') as f:
            loaded_model = model_from_yaml(f.read())
            loaded_model.load_weights(mf+".h5")
            Log(LOG_INFO) << "DNN model loaded from %s.yaml & .h5" % mf
            return loaded_model
    else:
        Log(LOG_FATAL) << "%s not supported" % ext    
            
    return None
    
