'''
Created on Sep 8, 2018

@author: fxua
'''
from keras.models import Sequential
from keras.layers import Dense

def createDNNModel(input_dim, neurons, init_wt, act, end_act, loss, optm_algo):
    model = Sequential()
    for n in range(len(neurons)):
        if n == 0:
            model.add(Dense(neurons[n], input_dim = input_dim, kernel_initializer=init_wt, activation = act[n]))
        else:
            model.add(Dense(neurons[n], kernel_initializer=init_wt, activation = act[n]))
    
    model.add(Dense(1, kernel_initializer=init_wt, activation = end_act))
    model.compile(loss = loss, optimizer = optm_algo, metrics=['accuracy'])

    return model
