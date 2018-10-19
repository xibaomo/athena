'''
Created on Sep 8, 2018

@author: fxua
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers

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
