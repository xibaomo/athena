import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_elements(arr, n, m):

  orig_arr = arr.copy()   # Make copy of original array
  orig_len = len(orig_arr)

  remove_idx = [i for i in range(orig_len) if i%n==0]

  for idx in remove_idx[::-1]:

    # Slice original array using fixed indices
    arr = np.concatenate((arr[:idx], arr[idx+m:]))

  return arr

# Create an example array with 1000 points
df = pd.read_csv('online.csv')
ask = df['ask_rtn'].values
# bid = df['bid_rtn'].values

# Create an array of indices where the vertical lines should be placed
vertical_lines_indices = np.arange(0, len(ask), 288)

# Plot the array
# ask = remove_elements(ask,288,24)
pos_ask = ask[ask<=0]
plt.plot(ask, '*-')
# plt.plot(bid, '*-')

# Add vertical lines at the specified indices
for index in vertical_lines_indices:
    plt.axvline(index, color='red', linestyle='dashed')

# Set plot labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot with Vertical Lines at Every 288 Points')

# Show the plot
plt.show()

