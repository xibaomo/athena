import robin_stocks.robinhood as rh
import sys,os
from utils import *
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns

BARS_PER_DAY=1

if len(sys.argv) < 2:
    print("Usage: {} <ticker>".format(sys.argv[0]))
    sys.exit(1)

username = os.getenv("BROKER_USERNAME")
password = os.getenv("BROKER_PASSWD")
rh.login(username, password)

ticker = sys.argv[1]

df = download_from_robinhood(ticker,interval='day',span='year')
rtns = df['Close'].pct_change().values
lookback=22*3*BARS_PER_DAY
lookfwd = 22*BARS_PER_DAY

ys = []
max_ks = 0
for i in range(lookback,len(rtns)-lookfwd):
    t1 = rtns[-lookback+i:i]
    t2 = rtns[i:i+lookfwd]
    ks,pval = stats.ks_2samp(t1,t2)
    ys.append(pval)
    if ks > max_ks:
        max_ks = ks
        idx = i

print(np.mean(ys))
print("max_ks: ", max_ks)
# mid = idx
t1=rtns[idx-lookback:idx]
t2=rtns[idx:idx+lookfwd]
# sns.ecdfplot(t1, label='Empirical CDF', color='blue')
# sns.ecdfplot(t2, label='Empirical CDF', color='blue')
# plt.show()
plt.plot(ys,'.')
plt.show()