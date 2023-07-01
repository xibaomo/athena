import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical price data for EUR/USD
data = yf.download("EURUSD=X", period="24mo", interval="1d")

# Compute daily returns
data['DailyReturn'] = data['Close'].pct_change()

# Compute monthly standard deviation of daily returns
monthly_std_dev = data['DailyReturn'].resample('M').std()

# Compute 5-day moving average of monthly standard deviation
monthly_std_dev_ma = monthly_std_dev.rolling(window=5).mean()

# Plot the standard deviation and moving average
plt.figure(figsize=(10, 6))
plt.plot(monthly_std_dev.index, monthly_std_dev, label='Standard Deviation')
plt.plot(monthly_std_dev_ma.index, monthly_std_dev_ma, label='Moving Average (5-day)')
plt.xlabel('Month')
plt.ylabel('Standard Deviation')
plt.title('Monthly Standard Deviation and Moving Average')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show(block=False)
