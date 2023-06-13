#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 01:15:35 2022

@author: naopc
"""

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import numpy as np
import sys, os
from sklearn.preprocessing import *
mkv_path = os.environ['ATHENA_HOME']+'/py_algos/fex_tf'
sys.path.append(mkv_path)
from fexconf import *
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

class SaveBestModel(Callback):
    def __init__(self, save_path, train_weight):
        super(SaveBestModel, self).__init__()
        self.save_path = save_path
        self.best_accuracy_sum = -float('inf')
        self.tw = train_weight

    def on_epoch_end(self, epoch, logs = None):
        accuracy_sum = logs.get('accuracy')*self.tw + logs.get('val_accuracy')*(1-self.tw)
        if accuracy_sum > self.best_accuracy_sum:
            self.best_accuracy_sum = accuracy_sum
            self.model.save(self.save_path)
            print('Saved best model with accuracy sum:', accuracy_sum)

fexconf = FexConfig(sys.argv[1])

fm = np.load(fexconf.getFeatureFile())
labels = np.load(fexconf.getLabelFile())

#ffm = fm[:, :]
#ffm = fm[:, 2:]
#ffm = fm[:, [0, 1, 2, 3, 4, 5, 6, 7   ]]
N = fexconf.getAllDataSize()
end_pos = fexconf.getDataEndPos()
if end_pos is None:
    start_pos = N
else:
    start_pos = end_pos+N

ffm = fm[-start_pos:-end_pos, [0, 1, 2, 6  ]]

flbs = labels[-start_pos:-end_pos]

valid_size = fexconf.getValidSize()
test_size = fexconf.getTestSize()

if ( test_size +valid_size> ffm.shape[0]):
    print("Error: all data size: {}, test size: {}".format(fm.shape[0], test_size))
    sys.exit(1)
x_train = ffm[:-test_size-valid_size, :]
x_valid = ffm[-test_size-valid_size:-test_size, :]
x_test = ffm[-test_size:, :]
y_train = flbs[:-test_size-valid_size].astype(np.uint8)-1
y_valid = flbs[-test_size-valid_size:-test_size].astype(np.uint8)-1
y_test = flbs[-test_size:].astype(np.uint8)-1

print("train size: ",x_train.shape[0])
print("valid size: ",x_valid.shape[0])
print("test size: ",x_test.shape[0])
scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_valid = scaler.transform(x_valid)

s = x_train.shape[1]
ns = 16*8*8
drpr = 0.25
model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(1, 2)),
  tf.keras.layers.Dense(ns*1, input_shape=(s, ), activation='relu',kernel_regularizer = regularizers.l2(0.001)),
  tf.keras.layers.Dropout(drpr),
#    tf.keras.layers.Dense(ns, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
#    tf.keras.layers.Dropout(drpr),

    #tf.keras.layers.Dense(ns, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
    #tf.keras.layers.Dropout(drpr),

    tf.keras.layers.Dense(ns/2, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
   tf.keras.layers.Dropout(drpr),

tf.keras.layers.Dense(ns/16, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
   tf.keras.layers.Dropout(drpr),

  tf.keras.layers.Dense(2, activation='softmax')
])
# predictions = model(x_train[:1]).numpy()

#checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only = True, mode='max')
tw = fexconf.getTrainWeight()
mf = fexconf.getModelFile()
checkpoint = SaveBestModel(mf, tw)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs = 400,
        batch_size = 128*2,
        validation_data=(x_valid, y_valid),
        callbacks=[checkpoint])

print("Checking performance of the best model: ")
best_model = tf.keras.models.load_model(mf)
_, train_acc = best_model.evaluate(x_train, y_train, verbose = 2)
_, valid_acc = best_model.evaluate(x_valid, y_valid, verbose = 2)
score = train_acc*tw + valid_acc*(1-tw)
print(f"Highest score: {score:.2f}")
print("Accuracy on test set: ")
best_model.evaluate(x_test, y_test, verbose = 2)

yp = best_model.predict(x_test)
yp = tf.nn.softmax(yp)
y_pred = tf.argmax(yp, 1).numpy()
print(y_pred)
print('y_train dist: ',np.sum(y_train==y_train[0])/len(y_train))
print('y_test dist: ',np.sum(y_test==y_test[0])/len(y_test))

#tf.saved_model.save(best_model, 'saved_model')
