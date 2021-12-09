#!/usr/bin/env python

import pandas as pd

csvfile = "features.csv"
keys = ['DMA','RSI','ROC','EMA','KAMA','LAG']
df = pd.read_csv(csvfile)

gp = df.groupby(keys)
lgp = list(gp)

print "all gauges: ",df.shape[0]
print "unique gauges: ",len(lgp)

for g in lgp:
    if g[1].shape[0] > 1:
        print g[1]

