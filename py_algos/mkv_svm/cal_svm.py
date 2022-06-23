#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:20:35 2022

@author: naopc
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
MIN_SPEED = 4.5E-6
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
def plot_labels(ffm,flbs):
    for i in range(len(flbs)):
        if flbs[i] == 0:
            plt.plot(ffm[i,0],ffm[i,1],'gs')
        if flbs[i] == 1:
            plt.plot(ffm[i,0],ffm[i,1],'ro',fillstyle='none')
        if flbs[i] == 2:
            plt.plot(ffm[i,0],ffm[i,1],'bx')
        if flbs[i] == 3:
            plt.plot(ffm[i,0],ffm[i,1],'d')
            
def eval_model(model, x_test, y_test):
    yy = model.predict(x_test)
    res = (yy==y_test)
    acc = sum(res)/len(res)
    print("accuracy: ",acc)
    
if __name__ == "__main__":
    fm = np.load('fm.npy')
    labels = np.load('labels.npy')
    
    idx = fm[:,1] >= MIN_SPEED
    
    ffm = fm[idx,:]
    flbs = labels[idx]
    
    # idx = ffm[:,0] <0.5
    # ffm = ffm[idx,:]
    # flbs = flbs[idx]
    
    scaler = StandardScaler()
    ffm = scaler.fit_transform(ffm)
    # ffm[:,1] = ffm[:,1]*1e5
    
    plot_labels(ffm, flbs)
    
    clf = svm.SVC(kernel='rbf', C=1e3)
    clf.fit(ffm,flbs)
    pickle.dump(clf, open("svm.pkl", 'wb'))
    print("model saved: svm.pkl")
    plot_svc_decision_function(clf)
    
    eval_model(clf,ffm,flbs)
    plt.show()
    
    
    