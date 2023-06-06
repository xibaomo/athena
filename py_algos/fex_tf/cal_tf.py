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

mkvconf = FexConfig(sys.argv[1])

fm = np.load(mkvconf.getFeatureFile())
labels = np.load(mkvconf.getLabelFile())

#ffm = fm[:, :]
#ffm = fm[:, 2:]
#ffm = fm[:, [0, 1, 2, 3, 4, 5, 6, 7   ]]
ffm = fm[:, [0, 1, 2, 3, 4, 5   ]]
flbs = labels

test_size = int(200)

if ( test_size > ffm.shape[0]):
    print("Error: all data size: {}, test size: {}".format(fm.shape[0], test_size))
    sys.exit(1)
x_train = ffm[:-test_size, :]
x_test = ffm[-test_size:, :]
y_train = flbs[:-test_size].astype(np.uint8)-1
y_test = flbs[-test_size:].astype(np.uint8)-1

print("train size: ",x_train.shape[0])
print("test size: ",x_test.shape[0])
scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

s = x_train.shape[1]
ns = 16*8*8
drpr = 0.2
model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(1, 2)),
  tf.keras.layers.Dense(ns*2, input_shape=(s, ), activation='relu',kernel_regularizer = regularizers.l2(0.001)),
  tf.keras.layers.Dropout(drpr),
    tf.keras.layers.Dense(ns, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
    tf.keras.layers.Dropout(drpr),

    tf.keras.layers.Dense(ns, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
    tf.keras.layers.Dropout(drpr),

    tf.keras.layers.Dense(ns/2, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
   tf.keras.layers.Dropout(drpr),

tf.keras.layers.Dense(ns/4, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
   tf.keras.layers.Dropout(drpr),

  tf.keras.layers.Dense(2, activation='softmax')
])
# predictions = model(x_train[:1]).numpy()

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only = True, mode='max')
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs = 200,
        batch_size = 128*4,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint])
best_val_accuracy = max(history.history['val_accuracy'])
print(f"Best accuracy on validation set: {best_val_accuracy}")

best_model = tf.keras.models.load_model('best_model.h5')
best_model.evaluate(x_train, y_train, verbose = 2)
best_model.evaluate(x_test, y_test, verbose = 2)

yp = best_model.predict(x_test)
yp = tf.nn.softmax(yp)
y_pred = tf.argmax(yp, 1).numpy()
print(y_pred)
print('y_train dist: ',np.sum(y_train==y_train[0])/len(y_train))
print('y_test dist: ',np.sum(y_test==y_test[0])/len(y_test))

#tf.saved_model.save(best_model, 'saved_model')
