import pdb
import pandas as pd
import sys,os
import networkx as nx
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

def getNodes(all_pairs):
    nodes=[]
    for p in all_pairs:
        s1 = p[:3]
        if not s1 in nodes:
            nodes.append(s1)
        s2 = p[3:]
        if not s2 in nodes:
            nodes.append(s2)
    return nodes
def add_days_to_date(date_str, num_days):
    # Convert string to datetime object
    # pdb.set_trace()
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Add the specified number of days to the date
    new_date = date + timedelta(days=num_days)

    # Convert the resulting date back to string format
    new_date_str = new_date.strftime('%Y-%m-%d')

    return new_date_str
def createGraph(nodes,prices):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in nodes:
        G.add_node(node)

    for col in prices.keys():
        # pdb.set_trace()
        s1 = col[:3]
        s2 = col[3:]
        if not np.isnan(prices[col]):
            G.add_edge(s1,s2,weight=np.log(prices[col]))
            G.add_edge(s2,s1,weight=-np.log(prices[col]))

    return G

def computeLimitRtns(G,src_node,tar_node):
    # pdb.set_trace()
    all_paths = list(nx.all_simple_paths(G, source=src_node, target=tar_node))

    pair = tar_node + src_node
    if not pair in prices.keys():
        pair = src_node + tar_node
        w = -np.log(prices[pair])
    else:
        w = np.log(prices[pair])

    wts = []
    for path in all_paths:
        total_weight = sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)) + w
        wts.append(total_weight)
        # if abs(total_weight) > 0e-4:
        #     print(" -> ".join(path), f"Total Weight: {total_weight}")
    wts = np.array(wts)
    mx = np.max(wts)
    mi = np.min(wts)
    score  = mx if abs(mx) >= abs(mi) else mi
    return mi,mx,score

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: {} <forex_list> <date> <history_days>'.format(sys.argv[0]))
        sys.exit(1)

    all_pairs = pd.read_csv(sys.argv[1],comment='#')['<SYM>'].values
    nodes = getNodes(all_pairs)
    nodes = ['USD','EUR','JPY','CAD','AUD','NZD','GBP','CHF']
    end_date = sys.argv[2]
    past_days = int(sys.argv[3])
    start_date = add_days_to_date(end_date,-past_days)

    syms = all_pairs + '=X'
    data = yf.download(syms.tolist(), start = start_date, end = end_date,interval='1h')['Close']
    # data = data.dropna(axis=1)
    syms = data.keys().tolist()
    df = pd.DataFrame()
    for i in range(len(syms)):
        syms[i] = syms[i][:-2]
    data.columns = syms
    for col in data.keys():
        if col[:3] in nodes and col[3:] in nodes:
            df[col] = data[col]
    # pdb.set_trace()
    prices = df.iloc[3,:]
    G = createGraph(nodes,prices)

    for sym in nodes:
        if sym == 'USD':
            continue
        mi,mx,_ = computeLimitRtns(G,'USD',sym)
        print("USD->{}: min: {:.4e}, max: {:.4e}".format(sym,mi,mx))

