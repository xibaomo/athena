from glpconf import GraphloopConfig
from graphloop import createGraph, computeLimitRtns, getTruePair
import pandas as pd
import sys, os
import numpy as np
import pdb
glpconf = None
all_pairs = []
def generateTradeSyms(nodes, all_forex):
    pairs = []
    for sym in all_forex:
        s1 = sym[:3]
        s2 = sym[3:]
        if s1 in nodes and s2 in nodes:
            pairs.append(sym)
    return pairs

def generateTradePairs(path, loop_pos_type):
    global all_pairs
    syms = []
    pos_type = []
    for i in range(len(path)-1):
        edge = [path[i], path[i+1]]
        sym, isFlip = getTruePair(edge, all_pairs)
        syms.append(sym)
        if not isFlip:
            pos_type.append(loop_pos_type)
        else:
            pos_type.append(-loop_pos_type)

    return syms, pos_type

def init(config_file):
    global glpconf, all_pairs
    glpconf =GraphloopConfig(config_file)

    forex_df = pd.read_csv(glpconf.getForexListFile(), comment='#')
    nodes = glpconf.getSelectedNodes()

    all_pairs = generateTradeSyms(nodes, forex_df['<SYM>'])
    return all_pairs
def process(prices):
    global glpconf, all_pairs
    # df_empty = pd.DataFrame(columns=all_pairs)
    new_row = pd.DataFrame(prices, columns=all_pairs)
    # prices_df = pd.concat([df_empty, new_row], ignore_index=True)
    # pdb.set_trace()
    G = createGraph(glpconf.getSelectedNodes(), new_row.iloc[0, :])
    high_score, high_path, low_score, low_path = computeLimitRtns(G, 'USD',glpconf.getEndNode(),new_row.iloc[0,:])

    pairs, pos_type = generateTradePairs(high_path, -1)

    return pairs, pos_type
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} <mtn.yaml>'.format(sys.argv[0]))
        sys.exit(1)

    pairs = init(sys.argv[1])
    print("all trade pairs: ", pairs, len(pairs))

    prices = np.random.random((1, 28))
    syms, pos_type = process(prices)

    for s, p in zip(syms, pos_type):
        print(s, p)
