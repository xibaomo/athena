import pdb
import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Download historical price data for EUR/USD
df = pd.read_csv('forex_list.csv',comment='#')
syms = df['<SYM>'].values
syms = syms +"=X"
data = yf.download(syms.tolist(), period="6mo", interval="1d")
rtns = data['Close'].pct_change()

for sym in rtns.keys():
    r = rtns[sym].values[1:]
    res = adfuller(r)
    print(sym,res[1],np.mean(r))


sys.exit(0)

# Compute monthly average daily return
monthly_avg_return = data['DailyReturn'].resample('M').mean()

# Compute 5-day moving average of monthly average daily return
monthly_avg_return_ma = monthly_avg_return.rolling(window=5).mean()

# Plot the average daily return and moving average
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg_return.index, monthly_avg_return, label='Average Daily Return')
plt.plot(monthly_avg_return_ma.index, monthly_avg_return_ma, label='Moving Average (5-day)')
plt.xlabel('Month')
plt.ylabel('Return')
plt.title('Monthly Average Daily Return and Moving Average')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show(block=False)
