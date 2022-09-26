#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:20:35 2022

@author: naopc
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
mkv_path = os.environ['ATHENA_HOME']+'/py_algos/mkv_svm'
sys.path.append(mkv_path)
from mkvsvmconf import *
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

def plot_labels(ffm, flbs):
    for i in range(len(flbs)):
        if flbs[i] == 0:
            plt.plot(ffm[i, 0], ffm[i, 1], 'gs')
        if flbs[i] == 1:
            plt.plot(ffm[i, 0], ffm[i, 1], 'go',fillstyle='none')
        if flbs[i] == 2:
            plt.plot(ffm[i, 0], ffm[i, 1], 'rx')
        if flbs[i] == 3:
            plt.plot(ffm[i, 0], ffm[i, 1], 'd')

def eval_model(model, x_test, y_test):
    yy = model.predict(x_test)
    res = (yy==y_test)
    acc = sum(res)/len(res)
    print("accuracy: ",acc)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <mkv.yaml>".format(sys.argv[0]))
        sys.exit(1)

    mkvconf = MkvSvmConfig(sys.argv[1])

    fm = np.load(mkvconf.getFeatureFile())
    labels = np.load(mkvconf.getLabelFile())

    idx = fm[:, 1] > mkvconf.getMinSpeed()
    print("Samples in use:", sum(idx))
    ffm = fm[idx, :]
    lbs = labels[idx]
    scaler = StandardScaler()
    ffm = scaler.fit_transform(ffm)
    # ffm[:, 1] = ffm[:, 1]*1e5

    #plot_labels(ffm, flbs)

    clf = DecisionTreeClassifier(max_leaf_nodes = 9)
    clf.fit(ffm, lbs)
    mf = mkvconf.getModelFile()
    sf = mkvconf.getScalerFile()
    pickle.dump(clf, open(mf, 'wb'))
    print("model saved: ",mf)
    pickle.dump(scaler, open(sf, 'wb'))
    print("scaler saved: ",sf)

    eval_model(clf, ffm, lbs)
    tree.plot_tree(clf)
    plt.show()



