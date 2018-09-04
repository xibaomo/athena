'''
Created on Sep 3, 2018

@author: xfb
'''

from timeit import default_timer as timer

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