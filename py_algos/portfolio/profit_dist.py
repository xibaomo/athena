import pdb

import numpy as np
import matplotlib.pyplot as plt
import re
import sys, os
file_path = sys.argv[1] # Replace with the actual file path
cost_key = 'Final cost'
costs = []
sig_key = 'best std'
sigmas = []
profit_key = 'Actual profit'
profits = []
def __getVal(line):
    parts = line.split(':')
    # if len(parts) > 1:
    #     number = parts[1].strip()  # Get the number after ':'
    #     ps = number.split("\x1")
    #     val = ps[0]
    #     return float(val)
def getVal(line):
    parts = line.split(':')
    line = parts[1]
    pattern = r"[-+]?[0-9]*\.?[0-9]+"
    matches = re.findall(pattern, line)
    if matches:
        return float(matches[0])
# Open the file
with open(file_path, 'r') as file:
    # Read each line in the file
    for line in file:
        # Check if the line contains the key
        if cost_key in line:
            val = getVal(line)
            if abs(val) > 100:
                pdb.set_trace()
            costs.append(getVal(line))
        if sig_key in line:
            sigmas.append(getVal(line))

        if profit_key in line:
            # pdb.set_trace()
            val = getVal(line)
            if abs(val)> 10000:
                pdb.set_trace()
            profits.append(getVal(line))

sigmas = np.array(sigmas)
costs = np.array(costs)
profits = np.array(profits)
mp = max(abs(profits))
idx = np.argmin(costs)
print("min cost {} at sigma: {}, mu: {}".format(costs[idx],sigmas[idx],-sigmas[idx]*costs[idx]))
plt.scatter(sigmas, -costs*sigmas, c=profits, cmap='seismic',vmin=-mp,vmax=mp)
plt.plot(sigmas[idx],-sigmas[idx]*costs[idx],'*')
plt.colorbar()
plt.xlim(0, 0.005)
plt.figure()
plt.plot(sigmas,profits,'.')
plt.show()

