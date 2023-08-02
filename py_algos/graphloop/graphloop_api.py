from glpconf import GraphloopConfig
from graphloop import createGraph, computeLimitRtns, getTruePair
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
def process_quote(timestr, price_list):
    global glpconf, glp_box
    # df_empty = pd.DataFrame(columns=all_pairs)
    print("Processing...",timestr,price_list)

    all_syms = glp_box.all_syms
    print("creating dict...")
    data_dict = {all_syms[i]: price_list[i] for i in range(len(all_syms))}
    print("dict created")
    G = createGraph(glpconf.getSelectedNodes(), data_dict)
    print("graph created")

    if glp_box.path_list is None:
        glp_box.path_list = list(nx.all_simple_paths(G, source='USD', target=glpconf.getEndNode()))
    high_score, high_path, low_score, low_path = computeLimitRtns(G, glp_box.path_list,'USD',glpconf.getEndNode(),data_dict)

    print("max loop return: {:.4e}".format(high_score))

    if high_score < glpconf.getOpenPositionReturn():
        print("No action")
        return [],[]


    print(high_path)
    glp_box.current_loop = high_path
    syms, pos_type = generateTradeSyms(high_path, -1,glp_box.df.columns)

    return syms, pos_type
def get_loop():
    global glp_box
    return glp_box.current_loop
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} <mtn.yaml>'.format(sys.argv[0]))
        sys.exit(1)

    symlib = init(sys.argv[1])
    print("all trade pairs: ", symlib, len(symlib))

    prices = np.random.random(28).tolist()
    # prices = [0.94775, 0.64826, 91.955, 1.09646, 0.71346, 0.68353, 96.973, 141.742, 1.53908, 1.45956, 0.99842, 0.88728, 141.62,
    #  1.68852, 1.09873, 1.73342, 1.64364, 1.12399, 159.474, 1.9015, 1.23771, 0.86364, 0.59054, 83.78, 0.64996, 1.32827,
    #  0.90756, 128.886]
    syms, pos_type = process_quote("2023-07-28 01:00:00", prices)

    loop = get_loop()
    print("current loop: ",loop)

    for s, p in zip(syms, pos_type):
        print(s, p)
