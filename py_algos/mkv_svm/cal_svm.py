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
from sklearn.preprocessing import *
import pickle
mkv_path = os.environ['ATHENA_HOME']+'/py_algos/mkv_svm'
sys.path.append(mkv_path)
from mkvsvmconf import *
from pkl_predictor import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def plot_svc_decision_function(model, ax = None, plot_support = True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)

    xx, yy = np.meshgrid(x, y)
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.imshow(
        Z,
        interpolation='nearest',
        extent=(xx.min(), xx.max(), y.min(), yy.max()),
        aspect='auto',
        origin='lower',
        cmap = plt.cm.PuOr_r,
    )
    # plot decision boundary and margins
    ax.contour(xx, yy, Z, colors='k',
               levels=[-1, 0, 1], alpha = 0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s = 300, linewidth = 1, facecolors='none');
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

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
    pklconf = PklPredictorConfig(mkvconf)

    MIN_SPEED = float(pklconf.getMinSpeed())

    print("max speed: ", np.max(fm[:, 1]))
    print("min speed: ", MIN_SPEED)
    
    idx = ~np.isnan(fm[:,4]) 
    fm = fm[idx,:]
    labels = labels[idx]
    

    idx = fm[:, 1] >= MIN_SPEED
    # idx = fm[:, 1] >= 3e-6
    ffm = fm[idx, :]
    flbs = labels[idx]

    # for i in range(ffm.shape[0]):
    #     if ffm[i, 0] < 0.5:
    #         ffm[i, 1] = -ffm[i, 1]

    ffm = ffm[:, pklconf.getSelectedFeatureID()]

    test_size = 200

    if ( test_size > ffm.shape[0]):
        print("Error: all data size: {}, test size: {}".format(fm.shape[0], test_size))
        sys.exit(1)
    fm_train = ffm[:-test_size, :]
    fm_test = ffm[-test_size:, :]
    lb_train = flbs[:-test_size]
    lb_test = flbs[-test_size:]

    print("train size: ",fm_train.shape[0])
    print("test size: ",fm_test.shape[0])
    # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler = RobustScaler()
    fm_train = scaler.fit_transform(fm_train)

    #clf = svm.SVC(kernel='rbf', C = 1)
    #clf = DecisionTreeClassifier(max_leaf_nodes = 90, min_samples_leaf = 0.1)
    clf = RandomForestClassifier(n_estimators = 600)
    clf.fit(fm_train, lb_train)

    print("Computing accuracy on training set...")
    eval_model(clf, fm_train, lb_train)

    mf = pklconf.getModelFile()
    sf = pklconf.getScalerFile()
    pickle.dump(clf, open(mf, 'wb'))
    print("model saved: ",mf)
    pickle.dump(scaler, open(sf, 'wb'))
    print("scaler saved: ",sf)
    '''
    # plot_svc_decision_function(clf)
    '''

    xx = scaler.transform(fm_test)
    # xx = fm_test
    print("Computing accuracy on test set...")
    eval_model(clf, xx, lb_test)
    plt.show()



