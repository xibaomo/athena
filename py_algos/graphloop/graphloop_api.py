from glpconf import GraphloopConfig
from graphloop import createGraph, getTruePair, computeMinAskRtn
import pandas as pd
import sys, os
import numpy as np
import yaml
import networkx as nx
import pdb

class GlpBox(object):
    def __init__(self):
        self.df = None
        self.all_syms = None
        self.current_loop = None
        self.path_list = None
        self.min_ask_rtn = []

glp_box=None
glpconf = None

def generateSymLib(nodes, forex_list):
    symlib = []
    for sym in forex_list:
        s1 = sym[:3]
        s2 = sym[3:]
        if s1 in nodes and s2 in nodes:
            symlib.append(sym)
    return symlib

def generateTradeSyms(path, loop_pos_type, all_syms):
    syms = []
    pos_type = []
    for i in range(len(path)-1):
        src = path[i]
        dst = path[i+1]
        sym = src+dst
        if sym in all_syms:
            syms.append(sym)
            pos_type.append(loop_pos_type)
        else:
            sym = dst+src
            syms.append(sym)
            pos_type.append(-loop_pos_type)

    return syms, pos_type

def init(config_file):
    global glpconf,glp_box
    yamlroot = yaml.load(open(config_file), Loader=yaml.FullLoader)
    glpcf = yamlroot['GRAPHLOOP']['CONFIG_FILE']
    glpconf =GraphloopConfig(glpcf)

    forex_df = pd.read_csv(glpconf.getForexListFile(), comment='#')
    nodes = glpconf.getSelectedNodes()

    syms = generateSymLib(nodes, forex_df['<SYM>'])

    glp_box = GlpBox()
    glp_box.all_syms = syms.copy()
    glp_box.df = pd.DataFrame(columns=syms)
    del forex_df
    del yamlroot
    # pdb.set_trace()
    return syms
def process_quote(timestr, ask_list, bid_list):
    global glpconf, glp_box
    # df_empty = pd.DataFrame(columns=all_pairs)
    print("Processing...",timestr,ask_list)

    all_syms = glp_box.all_syms
    print("creating dict...")
    ask_dict = {all_syms[i]: ask_list[i] for i in range(len(all_syms))}
    bid_dict = {all_syms[i]: bid_list[i] for i in range(len(all_syms))}
    print("dict created")
    G = createGraph(glpconf.getSelectedNodes(),all_syms)
    print("graph created")

    if glp_box.path_list is None:
        path_list = list(nx.all_simple_paths(G, source='USD', target=glpconf.getEndNode()))
        glp_box.path_list = [path for path in path_list if len(path) > 2]
        for path in glp_box.path_list:
            path.append('USD')
    # high_score, high_path, low_score, low_path = computeLimitRtns(G, glp_box.path_list,'USD',glpconf.getEndNode(),ask_dict,bid_dict)

    min_rtn,opt_path = computeMinAskRtn(glp_box.path_list,ask_dict,bid_dict)
    print("min ask return: {:.4e}".format(min_rtn),opt_path)

    glp_box.min_ask_rtn.append(min_rtn)
    if min_rtn > glpconf.getBuyThresholdReturn():
    # if min_rtn < 9999:
        print("No action")
        return [],[]
    print(opt_path)
    glp_box.current_loop = opt_path
    syms, pos_type = generateTradeSyms(opt_path, 1,ask_dict.keys())

    return syms, pos_type
def get_loop():
    global glp_box
    return glp_box.current_loop

def finish():
    global glp_box
    data = np.array(glp_box.min_ask_rtn)
    df = pd.DataFrame(data,columns=['ask_rtn'])
    df.to_csv('online.csv')
    print("data dumped to online.csv")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} <mtn.yaml>'.format(sys.argv[0]))
        sys.exit(1)

    symlib = init(sys.argv[1])
    print("all trade pairs: ", symlib, len(symlib))

    # prices = np.random.random(28).tolist()
    ask = [0.94775, 0.64826, 91.955, 1.09646, 0.71346, 0.68353, 96.973, 141.742, 1.53908, 1.45956, 0.99842, 0.88728, 141.62,
     1.68852, 1.09873, 1.73342, 1.64364, 1.12399, 159.474, 1.9015, 1.23771, 0.86364, 0.59054, 83.78, 0.64996, 1.32827,
     0.90756, 128.886]
    bid = ask
    syms, pos_type = process_quote("2023-07-28 01:00:00", ask,bid)

    loop = get_loop()
    print("current loop: ",loop)

    for s, p in zip(syms, pos_type):
        print(s, p)

    finish()