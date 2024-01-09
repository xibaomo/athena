import pdb

import numpy as np
import matplotlib.pyplot as plt
import re
import sys, os
file_path = sys.argv[1] # Replace with the actual file path
cost_key = 'Final cost'
costs = []
mu_key = 'best mean'
mu = []
sig_key = 'best std'
sigmas = []
profit_key = 'Actual profit'
profits = []
hprof = []
lprof =[]
def extract_prices(string):
    # pdb.set_trace()
    prices = re.findall(r'\$-?\d+(?:\.\d{1,2})?', string)
    return [float(price[1:]) for price in prices]
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
    pattern = r"[-+]?[0-9]*\.?[0-9]+[e]?[+-]?\d*"
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
            # if abs(val) > 100:
            #     pdb.set_trace()
            costs.append(getVal(line))
        if mu_key in line:
            v = getVal(line)
            # if v > 0.001:
            #     pdb.set_trace()
            mu.append(getVal(line))
        if sig_key in line:
            sigmas.append(getVal(line))

        if "Profits(" in line:
            # pdb.set_trace()
            pcs = extract_prices(line)
            hprof.append(pcs[1])
            lprof.append(pcs[0])
        if profit_key in line:
            # pdb.set_trace()
            val = getVal(line)
            # if abs(val)> 10000:
                # pdb.set_trace()
            profits.append(getVal(line))

mu = np.array(mu)
sigmas = np.array(sigmas)
costs = np.array(costs)
profits = np.array(hprof)
lprof = np.array(lprof)
hprof = np.array(hprof)
#filter out invalid costs
idx = costs < 0.05
costs = costs[idx]
mu = mu[idx]
sigmas=sigmas[idx]
profits = profits[idx]
lprof = lprof[idx]
hprof = hprof[idx]
mp = max(abs(profits))
idx = np.argmin(costs)
print(" low profit: [{:.2f},{:.2f},{:.2f}]".format(np.min(lprof),np.mean(lprof),np.max(lprof)))
print("high profit: [{:.2f},{:.2f},{:.2f}]".format(np.min(hprof),np.mean(hprof),np.max(hprof)))
plt.scatter(sigmas, costs, c=lprof, cmap='seismic',vmin=-mp,vmax=mp)
# plt.plot(sigmas[idx],-sigmas[idx]*costs[idx],'y*')
plt.colorbar()
plt.figure()
plt.scatter(sigmas, costs, c=hprof, cmap='seismic',vmin=-mp,vmax=mp)
plt.colorbar()
plt.figure()
plt.scatter(sigmas, mu, c=hprof, cmap='seismic',vmin=-mp,vmax=mp)
plt.colorbar()
# plt.xlim(0, 0.005)
# plt.figure()
# plt.plot(sigmas,profits,'.')
plt.show()

