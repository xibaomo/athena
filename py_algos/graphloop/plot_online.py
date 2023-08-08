import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create an example array with 1000 points
df = pd.read_csv('online.csv')
ask = df['ask_rtn'].values
bid = df['bid_rtn'].values

# Create an array of indices where the vertical lines should be placed
vertical_lines_indices = np.arange(0, len(ask), 288)

# Plot the array
plt.plot(ask, '*-')
plt.plot(bid, '*-')

# Add vertical lines at the specified indices
for index in vertical_lines_indices:
    plt.axvline(index, color='red', linestyle='dashed')

# Set plot labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot with Vertical Lines at Every 288 Points')

# Show the plot
plt.show()

