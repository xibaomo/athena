import pdb

import tensorflow as tf
import numpy as np
from logger import *
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='accuracy', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()

class DNNClassifier(object):
    def __init__(self,cfg,x_dim,y_dim):
        self.config = cfg
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128,activation='relu',input_shape=x_dim),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(y_dim)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        self.model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

        print(self.model.summary())

    def fit(self,x_train,y_train):
        self.save_best_model = SaveBestModel()
        epochs = self.config.getDNNEpochs()

        self.model.fit(x_train,y_train,epochs = epochs,verbose=1, callbacks=[self.save_best_model])

        # self.model.set_weights(self.save_best_model.best_weights)
        self.model.evaluate(x_train, y_train, verbose = 2)

    def predict(self,x):
        prob_model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])
        raw_y = prob_model(x).numpy()
        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = np.argmax(raw_y[i])

        return y

    def save(self):
        mf = self.config.getTFModelFile()
        Log(LOG_WARNING) << "TODO: save tf model"

class CNNClassifier(DNNClassifier):
    def __init__(self,cfg,x_dim,y_dim):
        if len(x_dim) == 1:
            x_dim = x_dim + (1,)
        self.config = cfg
        ks = 10
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1., input_shape=x_dim),
            tf.keras.layers.Conv1D(16,ks,padding='same',activation='relu'),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Conv1D(32, ks, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Conv1D(64, ks, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(y_dim)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

        print(self.model.summary())
