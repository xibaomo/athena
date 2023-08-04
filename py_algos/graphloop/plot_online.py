import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create an example array with 1000 points
df = pd.read_csv('online.csv')
data_array = df['ask_rtn'].values

# Create an array of indices where the vertical lines should be placed
vertical_lines_indices = np.arange(0, len(data_array), 288)

# Plot the array
plt.plot(data_array, '*-')

# Add vertical lines at the specified indices
for index in vertical_lines_indices:
    plt.axvline(index, color='red', linestyle='dashed')

# Set plot labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot with Vertical Lines at Every 288 Points')

# Show the plot
plt.show()

