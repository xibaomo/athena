'''
Created on Sep 3, 2018

@author: xfb
'''

from timeit import default_timer as timer
import csv
import sys
from modules.basics.common.logger import LOG_FATAL, LOG_INFO

def tic():
    return timer()

def toc(tic):
    print "Elapsed time = %f s" % (timer() - tic)
    
def getAllYamlKeys(ymlTree,level_prefix,allKeys=[]):
    for k,v in ymlTree.items():
        key = level_prefix + "/" + k 
        if not isinstance(v, dict):
            allKeys.append(key)
        else:
            getAllYamlKeys(v, key, allKeys)
            
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
        
                
            
            
