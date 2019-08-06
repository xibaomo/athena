#!/usr/bin/env python

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import numpy as np

def coint_verify(list_x, list_y) :
    x = np.array(list_x)
    y = np.array(list_y)

    af = adfuller(np.diff(x))
    if af[1] > 0.01:
        print "diff(x) is not steady, p value = %f" % af[1]
        return 0
    af = adfuller(np.diff(y))
    if af[1] > 0.01:
        print "diff(y) is not steady, p value = %f" % af[1]
        return 0

    c = coint(x, y)
    if c[1] > 0.01:
        print "cointegration test fails. p value = %f" % c[1]
        return 0

    return 1

def test_adf(list_x):
    af = adfuller(np.array(list_x))
    return af[1]

def test_test(lx, ly):
    print lx.shape
    print ly.shape

if __name__ == "__main__":

    import random
    lx=[random.uniform(1, 10) for i in range(100)]
    ly=[random.uniform(1, 10) for i in range(100)]

    print coint_verify(lx, ly)
