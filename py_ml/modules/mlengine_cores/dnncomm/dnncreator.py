'''
Created on Sep 8, 2018

@author: fxua
'''
import keras.initializers as initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.models import model_from_yaml
from modules.basics.conf.generalconf import gGeneralConfig
from modules.mlengine_cores.dnncomm.dnnconf import DNNConfig
from keras import optimizers
import os
from modules.basics.common.logger import *
import pdb
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

Regularizer_switcher = {
    'L1': regularizers.l1,
    'L2': regularizers.l2,
    'L1L2': regularizers.l1_l2
    }

Optimizer_switcher = {
    "SGD": optimizers.SGD,
    "RMSprop": optimizers.RMSprop,
    "Adagrad": optimizers.Adagrad,
    "Adadelta": optimizers.Adadelta,
    "Adam": optimizers.Adam,
    "Adamax": optimizers.Adamax,
    "Nadam": optimizers.Nadam
    }

def createDNNModel(input_dim, end_act, loss, outnodes=1):
    config =  DNNConfig()
    neurons = config.getNeurons()
    init_wt = config.getWeightInit()
    dropouts = config.getDropoutRate()
    regs = config.getRegularizer()
    act = config.getActivation()
    optm_algo = config.getAlgorithm()
    lr = config.getLearnRate()
    mom = config.getMomentum()
    
    model = Sequential()

    # add two layers to standardize
    model.add(Dense(input_dim,input_dim=input_dim,name='offset',
                    trainable=False,
                    kernel_initializer = initializers.Identity(gain=1.)))
    model.add(Dense(input_dim,input_dim=input_dim,name='scaler',
                    trainable=False,
                    kernel_initializer = initializers.Identity(gain=1.)))

    for n in range(len(neurons)):
        reg,rv=regs[n].split(":")
        regler = Regularizer_switcher[reg]
        if n == 0:
            model.add(Dense(neurons[n], input_dim = input_dim, 
                            kernel_initializer=init_wt, 
                            kernel_regularizer=regler(float(rv))
                            ))
        else:
            model.add(Dense(neurons[n], kernel_initializer=init_wt,
                            kernel_regularizer=regler(float(rv)) 
                            ))
        
        model.add(BatchNormalization(axis=1))
        model.add(Activation(act[n]))
        if dropouts[n] > 0.0:
            model.add(Dropout(dropouts[n]))
    if lr < 0:
        optm = Optimizer_switcher[optm_algo]()
    else:
        if optm_algo == "SGD":
            optm = Optimizer_switcher[optm_algo](lr=lr,momentum=mom)
        else:
            optm = Optimizer_switcher[optm_algo](lr=lr)
            
    model.add(Dense(outnodes, kernel_initializer=init_wt, activation = end_act,name='output'))
    model.compile(loss = loss, optimizer = optm, metrics=['accuracy'])

    model.summary()
    return model

def saveDNNModel(model,modelfile):
#     pdb.set_trace()
    if modelfile.find('.yaml') < 0:
        modelfile = modelfile+'.yaml'
        
    mf,ext = os.path.splitext(modelfile)
    if ext == ".yaml" or ext == ".yml":
        model_yaml = model.to_yaml()
        with open(modelfile,'w') as f:
            f.write(model_yaml)
        model.save_weights(mf+".h5")
        
        Log(LOG_INFO) << "Model saved to %s.yaml & .h5" % mf
    elif ext == ".json":
        model_json = model.to_json()
        with open(mf+".json",'w') as json_file:
            json_file.write(model_json)

        model.save_weights(mf+".h5")

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
    elif ext == ".json":
        json_file = open(mf+'.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(mf+".h5")
        return loaded_model
    else:
        Log(LOG_FATAL) << "%s not supported" % ext    
            
    return None
    
