from glpconf import GraphloopConfig
from graphloop import *
import pandas as pd
import sys, os
import numpy as np
import yaml
import networkx as nx
import time
import pdb

class GlpBox(object):
    def __init__(self):
        self.df = None
        self.all_syms = None
        self.current_loop = None
        self.path_list = None
        self.min_ask_rtns = []
        self.max_bid_rtns = []
        self.trade_times = []

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
def generatePriceLot(syms,pos_type,ask_dict,bid_dict):
    price=[]
    lot = []
    for sym,pt in zip(syms,pos_type):
        p=0.
        if pt > 0:
            p = ask_dict[sym]
        else:
            p = bid_dict[sym]
        price.append(p)
        if sym[:3] == 'EUR':
            lot.append(1.)
            continue
        s = 'EUR' + sym[:3]
        if s in ask_dict.keys():
            lot.append(ask_dict[s])
        else:
            s = sym[:3]+'EUR'
            lot.append(1./ask_dict[s])
    return price,lot

def compOptPath(ask_dict,bid_dict):
    global glp_box,glpconf
    if glp_box.path_list is None:
        all_syms = ask_dict.keys()
        G = createGraph(glpconf.getSelectedNodes(), all_syms)
        print("graph created")
        path_list = list(nx.all_simple_paths(G, source='USD', target=glpconf.getEndNode()))
        glp_box.path_list = [path for path in path_list if len(path) > 2]
        for path in glp_box.path_list:
            path.append('USD')
    # high_score, high_path, low_score, low_path = computeLimitRtns(G, glp_box.path_list,'USD',glpconf.getEndNode(),ask_dict,bid_dict)

    min_rtn,opt_path = computeMinMidRtn(glp_box.path_list,ask_dict,bid_dict)
    return opt_path,min_rtn

def __compOptPath(ask_dict,bid_dict):
    global glpconf,glp_box
    mid_dict = {}
    for key in ask_dict.keys():
        p = (ask_dict[key] + bid_dict[key]) * .5
        mid_dict[key] = p
    vals = []
    for k,v in mid_dict.items():
        w = np.log(v)
        vals.append(w)
        vals.append(-w)
    base = min(vals)

    G = createGraphWeight(glpconf.getSelectedNodes(),mid_dict.keys(),mid_dict,base)
    # opt_path = nx.bellman_ford_path(G, 'USD', 'EUR')
    # min_rtn = nx.bellman_ford_path_length(G,'USD','EUR') + np.log(mid_dict['EURUSD'])

    source_node = 'USD'
    target_node = 'EUR'
    w = G[source_node][target_node]['weight']
    G[source_node][target_node]['weight']=999
    print('finding shortest path from {} to {}...'.format(source_node,target_node))
    # path = nx.algorithms.shortest_paths.dijkstra_path(G,source_node,target_node)
    _,path = nx.algorithms.shortest_paths.bidirectional_dijkstra(G, source_node, target_node)
    print('found')
    path.append('USD')
    G[source_node][target_node]['weight'] = w
    rtn = computePathMidRtn(path,mid_dict)
    return path,rtn

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
    parsed_time = pd.to_datetime(timestr)
    hour = parsed_time.hour
    # pdb.set_trace(
    all_syms = glp_box.all_syms
    print("creating dict...")
    ask_dict = {all_syms[i]: ask_list[i] for i in range(len(all_syms))}
    bid_dict = {all_syms[i]: bid_list[i] for i in range(len(all_syms))}
    print("dict created")

    tic = time.time()
    opt_path,min_rtn = compOptPath(ask_dict,bid_dict)
    print("finding shortest path takes {:.6f}".format(time.time()-tic))
    print("min return: {:.4e}".format(min_rtn),opt_path)

    # max_bid_rtn, opt_path = computeMaxBidRtn(glp_box.path_list, ask_dict, bid_dict)
    # print("max bid return: {:.4e}".format(max_bid_rtn), opt_path)

    glp_box.min_ask_rtns.append(min_rtn)
    # glp_box.max_bid_rtns.append(max_bid_rtn)
    if min_rtn > glpconf.getBuyThresholdReturn():
    # if min_ask_rtn < 9999:
        print("No action")
        return [],[]
    # print(opt_path)
    glp_box.trade_times.append(timestr)
    ask_rtn = computePathAskRtn(opt_path,ask_dict,bid_dict)
    print("ask rtn: {:.4e}".format(ask_rtn))
    glp_box.current_loop = opt_path
    syms, pos_type = generateTradeSyms(opt_path, 1,ask_dict.keys())

    price,lot = generatePriceLot(syms,pos_type,ask_dict,bid_dict)

    print("Trade info:")
    for sym,pos,pc,lz in zip(syms,pos_type,price,lot):
        print("{},{:2d},{:10.5f},{:.2f}".format(sym,pos,pc,lz))

    if not glpconf.isAllowPositions():
        return [],[],[],[]
    return syms, pos_type, price,lot
def get_loop():
    global glp_box
    return glp_box.current_loop

def finish():
    global glp_box,glpconf
    if glpconf.isAllowPositions():
        return
    data = np.array(glp_box.min_ask_rtns)
    df = pd.DataFrame(data,columns=['ask_rtn'])
    data = np.array(glp_box.max_bid_rtns)
    if len(data) > 0:
        df['bid_rtn'] = data
    df.to_csv('online.csv')
    print("data dumped to online.csv")

    data = np.array(glp_box.trade_times)
    df = pd.DataFrame(data,columns=['TRADE_TIME'])
    df.to_csv('trade_times.csv',index=False)
    print("trade times dumped to trade_times.csv")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} <mtn.yaml>'.format(sys.argv[0]))
        sys.exit(1)

    symlib = init(sys.argv[1])
    print("all trade pairs: ", symlib, len(symlib))

    # prices = np.random.random(28).tolist()
    ask = [0.90383, 0.61391, 94.74, 0.67844, 0.67833, 104.459, 153.821, 1.5925500000000001, 1.4414500000000001, \
           0.97777, 0.85903, 150.438, 1.07902, 1.85649, 1.67961, 1.1407, 175.891, 1.2572999999999999, 1.33386, 0.906, 139.319]
    bid = ask
    syms, pos_type, price, lot = process_quote("2023-07-28 05:00:00", ask,bid)

    syms, pos_type, price, lot = process_quote("2023-07-28 05:00:00", ask, bid)

    loop = get_loop()
    print("current loop: ",loop)

    for s, p,pc,lz in zip(syms, pos_type,price,lot):
        print(s, p, pc,lz)

    finish()