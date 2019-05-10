'''
Created on Sep 3, 2018

@author: xfb
'''

from timeit import default_timer as timer
import csv
import sys
from modules.basics.common.logger import *
import numpy as np
import random
import pandas as pd
import operator
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter 

def tic():
    return timer()

def toc(tic):
    print "Elapsed time = %f s" % (timer() - tic)
    
def getAllYamlKeys(ymlTree,level_prefix,allKeys=[]):
    for k,v in ymlTree.items():
        key = level_prefix + "/" + k 
        if not isinstance(v, dict) and v is not None:
            allKeys.append(key)
        elif isinstance(v,dict):
            getAllYamlKeys(v, key, allKeys)
        else:
            pass
            
    return

def findYamlKey(ymlTree,key):
    path=key.split("/")
    node = ymlTree
    for p in path:
        if p == "":
            continue
        v = node.get(p)
        if v == None:
            return None
        elif isinstance(v,dict):
            node = node[p] 
            continue
        else:
            return v 
        
    return

def updateYamlKey(ymlTree,key,value):
    path = key.split("/")
    node = ymlTree 
    for p in path:
        if p=="":
            continue
        v = node.get(p)
        if v == None:
            print "Key not found: %s" % key 
            return None
        elif isinstance(v, dict):
            node = node[p] 
            continue
        else:
            node[p] = value 
            return
    return

def cross_valid_split(fm,labels,test_ratio):
    ts = int(len(labels)*test_ratio)
    train_fm = fm[:ts+1,:]
    train_labels = labels[:ts+1]
    test_fm = fm[ts+1:,:]
    test_labels = labels[ts+1:]
    return train_fm,train_labels,test_fm,test_labels

class CSVParser(object):
    def __init__(self):
        return
    
    def load(self,filename,domKey):
        gaugeTable = {}
        f = open(filename,'rb')
        reader = csv.reader(f)
        headers = []
        k = 0
        for row in reader:
            if domKey in row:
                headers = row 
                continue
            
            if len(headers) != len(row):
                Log(LOG_FATAL) << "No. of fields inconsistent with headers"
                
            gauge = {}
            k+=1
            for m in range(len(headers)):
                gauge[headers[m]] = row[m]
                
            domvalue = gauge[domKey]
            if not gaugeTable.has_key(domvalue):
                gaugeTable[domvalue] = []
            
            gaugeTable[domvalue].append(gauge)
        
        Log(LOG_INFO) <<"%d gauges are loaded, %d unique gauges" % (k,len(gaugeTable))
        
        return gaugeTable
        
                

def computeRMS(diff):

    d2 = np.square(diff)
    rms = np.sqrt(np.mean(d2))
    return rms

def computeRange(errs):
    maxerr = 0.
    minerr = 0.
    for i in range(len(errs)):
        err = errs[i]
        if err > 0 and err > maxerr:
            maxerr = err
        if err < 0 and err < minerr:
            minerr = err

    return maxerr - minerr

def shuffleTwoList(fm,tg,seed=None):
    c = list(zip(fm,tg))
    random.Random(seed).shuffle(c)
    a,b = zip(*c)
    return np.vstack(a),np.array(b)

def smooth1D(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        Log(LOG_FATAL) << "Smooth only accepts 1d array"
        return
    if x.size < window_len:
        Log(LOG_FATAL) << "Array size should larger than window length"
        return
    if not window in ['flat','hanning','hamming','bartlett','blackman']:
        Log(LOG_FATAL) << "window type not supported"
        
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window=='flat':
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    
    return y

def savgol_smooth1D(lst,width,order):
#     print lst
    arr = np.array(lst)
    s =  savgol_filter(arr, width, order)
#     print "res = ",s
    return np.array(s)
