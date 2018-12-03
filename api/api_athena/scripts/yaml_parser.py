#!/usr/bin/env python

import yaml
import sys
import pdb
#sample key: EXTERNAL_MODEL/MODEL_PATH
def getValueByKey(key, yamlFile):
    yamlDict = yaml.load(open(yamlFile))

    key_chain = key.split('/')
    dt = yamlDict
    for k in key_chain:
        dt = dt[k]
        if not isinstance(dt, dict):
            return dt

if __name__ == "__main__":
    val = getValueByKey(sys.argv[1], sys.argv[2])
    print val

