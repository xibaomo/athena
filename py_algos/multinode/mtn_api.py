from mtnconf import MultinodeConfig
from multinode_arbitrage import createGraph,computeLimitRtns
import pandas as pd
import sys,os
import numpy as np
import pdb
mtnconf = None
all_pairs = []
def generateTradeSyms(nodes,all_forex):
    pairs = []
    for sym in all_forex:
        s1 = sym[:3]
        s2 = sym[3:]
        if s1 in nodes and s2 in nodes:
            pairs.append(sym)
    return pairs

def init(config_file):
    global mtnconf,all_pairs
    mtnconf = MultinodeConfig(config_file)

    forex_df = pd.read_csv(mtnconf.getForexListFile(),comment='#')
    nodes = mtnconf.getSelectedNodes()

    all_pairs = generateTradeSyms(nodes,forex_df['<SYM>'])
    return all_pairs
def process(prices):
    global mtnconf,all_pairs
    # df_empty = pd.DataFrame(columns=all_pairs)
    new_row = pd.DataFrame(prices, columns=all_pairs)
    # prices_df = pd.concat([df_empty, new_row], ignore_index=True)
    # pdb.set_trace()
    G = createGraph(mtnconf.getSelectedNodes(),new_row.iloc[0,:])
    high_score,high_path,low_score,low_path = computeLimitRtns(G,'USD',mtnconf.getEndNode(),new_row.iloc[0,:])
    return high_score,high_path
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} <mtn.yaml>'.format(sys.argv[0]))
        sys.exit(1)

    pairs = init(sys.argv[1])
    print("all trade pairs: ", pairs,len(pairs))

    prices = np.random.random((1,28))
    high_score,high_path = process(prices)

    print(high_score,high_path)
