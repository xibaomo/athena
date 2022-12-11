import ctypes as ct
import os
import pdb
import numpy as np

int32 = ct.c_int
uint64 = ct.c_uint64
real32 = ct.c_float
real64 = ct.c_double

libpath = "%s/release/lib/libathena_api.so" % os.environ['ATHENA_HOME']
_lib = ct.PyDLL(libpath)

try:
    _athena_minbar_label = _lib.athena_minbar_label
    _athena_minbar_label.argtypes = [ct.POINTER(real64), ct.POINTER(real64), ct.POINTER(real64), ct.POINTER(real64), ct.POINTER(real64), int32,
                                     ct.POINTER(int32), int32,
                                     real64,
                                     real64,
                                     int32,
                                     ct.POINTER(int32),
                                     ct.POINTER(int32)]
except:
    pass

'''
return labels. 1 - buy, 2 - sell, 0 - no action
'''

def minbar_label(op, hp, lp, cp, sp, time_ids, ret_thd, ret_ratio, max_stride):
    sp = 1.*sp
    num = time_ids.shape[0]
    labels = np.empty(num, dtype = int32)
    durations = np.empty(num, dtype = int32)
    _athena_minbar_label(op.ctypes.data_as(ct.POINTER(real64)),
                         hp.ctypes.data_as(ct.POINTER(real64)),
                         lp.ctypes.data_as(ct.POINTER(real64)),
                         cp.ctypes.data_as(ct.POINTER(real64)),
                         sp.ctypes.data_as(ct.POINTER(real64)),
                         op.shape[0],
                         time_ids.ctypes.data_as(ct.POINTER(int32)),
                         time_ids.shape[0],
                         ret_thd,
                         ret_ratio,
                         max_stride,
                         labels.ctypes.data_as(ct.POINTER(int32)),
                         durations.ctypes.data_as(ct.POINTER(int32)))
    return labels, durations

try:
    _athn_label_minbars = _lib.athn_label_minbars
    _athn_label_minbars.argtypes = [ct.POINTER(real64), ct.POINTER(real64), ct.POINTER(real64), ct.POINTER(uint64), int32,
                                     ct.POINTER(uint64), int32,
                                     real64,
                                     int32,
                                     ct.POINTER(int32)]
except:
    pass

def isSharpHour(tm):
    if tm.minute == 0 and tm.second == 0:
        return True
    return False
def athn_label_hours(df, stid, etid, ret_thd, ndays):
    op = df['<OPEN>'].values
    hi = df['<HIGH>'].values
    lw = df['<LOW>'].values
    tm = df['DATETIME']

    hourid=[]
    t0 = tm[0]
    secs = np.zeros(len(op), dtype = np.uint64)
    for i in range(stid, etid):
        if isSharpHour(tm[i]):
            hourid.append(i)
        dt = tm[i] - t0
        secs[i] = dt.total_seconds()
    hourid = np.array(hourid, dtype = np.uint64)

    labels = np.empty(len(hourid), dtype = int32)

    _athn_label_minbars(op.ctypes.data_as(ct.POINTER(real64)),
                        hi.ctypes.data_as(ct.POINTER(real64)),
                        lw.ctypes.data_as(ct.POINTER(real64)),
                        secs.ctypes.data_as(ct.POINTER(uint64)),
                        len(op),
                        hourid.ctypes.data_as(ct.POINTER(uint64)), len(hourid),
                        ret_thd,
                        ndays,
                        labels.ctypes.data_as(ct.POINTER(int32)))

    return hourid, labels




