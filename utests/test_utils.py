'''
Created on May 9, 2019

@author: fxua
'''
import modules.basics.common.utils as utils
import numpy as np


a = np.random.random(10)

lst = a.tolist()
print lst

s = utils.savgol_smooth1D(lst, 5, 3)
print s.shape