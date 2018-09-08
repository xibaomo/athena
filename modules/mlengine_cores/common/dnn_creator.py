from keras.models import Sequential
from keras.layers import Dense

def createDNNModel(input_dim, neurons, init_wt, act, end_act, loss, optm_algo):
    model = Sequential()
    for n in range(neurons):
        if n == 0:
            model.add(Dense(neurons[n], input_dim = input_dim, init = init_wt, activation = act[n]))
        else:
            model.add(Dense(neurons[n], init = init_wt, activation = act[n]))
        model.add(Dense(1, init = init_wt, activation = end_act))
        model.compile(loss = loss, optimizer = optm_algo, metrics=['accuracy'])

        return model
