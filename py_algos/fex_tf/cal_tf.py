#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 01:15:35 2022

@author: naopc
"""

import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
import sys, os
from sklearn.preprocessing import *
mkv_path = os.environ['ATHENA_HOME']+'/py_algos/mkv_svm'
sys.path.append(mkv_path)
from mkvsvmconf import *
from pkl_predictor import *

mkvconf = MkvSvmConfig(sys.argv[1])

fm = np.load(mkvconf.getFeatureFile())
labels = np.load(mkvconf.getLabelFile())
pklconf = PklPredictorConfig(mkvconf)

ffm = fm[:, 1:-1]
flbs = labels

test_size = 5

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
ns = 16
model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(1, 2)),
  tf.keras.layers.Dense(ns, input_shape=(s, ), activation='relu',kernel_regularizer = regularizers.l2(0.001)),
  tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(ns*2, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(ns*2, activation='relu',kernel_regularizer = regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(8, activation='relu'),
   tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)
])
# predictions = model(x_train[:1]).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer='adam',
              loss = loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 800)

model.evaluate(x_train, y_train, verbose = 2)
model.evaluate(x_test, y_test, verbose = 2)

yp = model.predict(x_test)
yp = tf.nn.softmax(yp)
y_pred = tf.argmax(yp, 1).numpy()
