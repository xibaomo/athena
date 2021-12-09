#!/usr/bin/env python

import numpy as np
import owls
import time
from scipy.stats import binom

n = 200
k = 100
p = 0.512
N = 1000

tic = time.time()
for i in range(N):
    a = owls.binom_pdf(k, n, p)
print time.time() - tic

tic = time.time()
for i in range(N):
    b = binom.pmf(k, n, p)
print time.time() - tic



