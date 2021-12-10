import ctypes as ct
import os
import numpy as np

int32 = ct.c_int
real32 = ct.c_float
real64 = ct.c_double

libpath = "%s/release/lib/libathena_api.so" % os.environ['ATHENA_HOME']
_lib = ct.PyDLL(libpath)

try:
    _athena_minbar_label = _lib.athena_minbar_label
    _athena_minbar_label.argtypes = [ct.POINTER(real64), ct.POINTER(real64), ct.POINTER(real64), ct.POINTER(real64), int32,
                                     ct.POINTER(int32), int32,
                                     real64,
                                     int32,
                                     ct.POINTER(int32)]
except:
    pass

'''
return labels. 1 - buy, -1 - sell, 0 - no action
'''

def minbar_label(op, hp, lp, cp, time_ids, ret_thd, max_stride):
    num = time_ids.shape[0]
    labels = np.empty(num, dtype = int32)
    _athena_minbar_label(op.ctypes.data_as(ct.POINTER(real64)),
                         hp.ctypes.data_as(ct.POINTER(real64)),
                         lp.ctypes.data_as(ct.POINTER(real64)),
                         cp.ctypes.data_as(ct.POINTER(real64)),
                         op.shape[0],
                         time_ids.ctypes.data_as(ct.POINTER(int32)),
                         time_ids.shape[0],
                         ret_thd,
                         max_stride,
                         labels.ctypes.data_as(ct.POINTER(int32)))
    return labels
