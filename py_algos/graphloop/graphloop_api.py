from glpconf import GraphloopConfig
from graphloop import createGraph, computeLimitRtns, getTruePair
import pandas as pd
import sys, os
import numpy as np
import pdb
glpconf = None
record_df = pd.DataFrame()
def generateSymLib(nodes, forex_list):
    symlib = []
    for sym in forex_list:
        s1 = sym[:3]
        s2 = sym[3:]
        if s1 in nodes and s2 in nodes:
            symlib.append(sym)
    return symlib

def generateTradeSyms(path, loop_pos_type,symlib):
    syms = []
    pos_type = []
    for i in range(len(path)-1):
        edge = [path[i], path[i+1]]
        sym, isFlip = getTruePair(edge, symlib)
        syms.append(sym)
        if not isFlip:
            pos_type.append(loop_pos_type)
        else:
            pos_type.append(-loop_pos_type)

    return syms, pos_type

def init(config_file):
    global glpconf,record_df
    glpconf =GraphloopConfig(config_file)

    forex_df = pd.read_csv(glpconf.getForexListFile(), comment='#')
    nodes = glpconf.getSelectedNodes()

    sym_lib = generateSymLib(nodes, forex_df['<SYM>'])

    record_df = pd.DataFrame(columns=sym_lib)
    return record_df.columns
def process(timestr, prices):
    global glpconf, record_df
    # df_empty = pd.DataFrame(columns=all_pairs)
    new_row = pd.DataFrame(prices, columns=record_df.columns)
    new_row['<TIME>'] = pd.to_datetime(timestr)
    new_row.set_index('<TIME>',inplace=True)

    record_df = pd.concat([record_df, new_row], ignore_index=False)
    # pdb.set_trace()
    G = createGraph(glpconf.getSelectedNodes(), new_row.iloc[0, :])
    high_score, high_path, low_score, low_path = computeLimitRtns(G, 'USD',glpconf.getEndNode(),new_row.iloc[0,:])

    syms, pos_type = generateTradeSyms(high_path, -1,record_df.columns)

    return syms, pos_type
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} <mtn.yaml>'.format(sys.argv[0]))
        sys.exit(1)

    symlib = init(sys.argv[1])
    print("all trade pairs: ", symlib, len(symlib))

    prices = np.random.random((1, 28))
    syms, pos_type = process("2023-07-28 01:00:00", prices)

    for s, p in zip(syms, pos_type):
        print(s, p)
