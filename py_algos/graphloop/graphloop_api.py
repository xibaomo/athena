from glpconf import GraphloopConfig
from graphloop import *
import pandas as pd
import sys, os
import numpy as np
import yaml
import networkx as nx
import time
import pdb
import statsmodels.api as sm
from scipy import stats

class GlpBox(object):
    def __init__(self):
        self.df = None
        self.all_syms = None
        self.current_loop = None
        self.current_loop_rtn = 0.
        self.path_list = None
        self.min_ask_rtns = []
        self.max_bid_rtns = []
        self.trade_times = {}
        self.sym2path_list={}
        self.cur_syms=None


glp_box=None
glpconf = None

def checkDep(x, y):
    id = y > np.median(y)
    x1 = x[id]
    x2 = x[~id]
    ks_statistic, p_value = stats.ks_2samp(x1, x2)
    return p_value

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

def __compOptPath(ask_dict,bid_dict):
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

def compOptPath(ask_dict,bid_dict):
    global glp_box,glpconf
    if len(glp_box.sym2path_list) == 0: # if this is the 1st run, find all loop paths
        all_syms = ask_dict.keys()
        G = createGraph(glpconf.getSelectedNodes(), all_syms)
        for sym in glpconf.getSelectedNodes():
            if sym == 'USD':
                continue
            paths = list(nx.all_simple_paths(G, source='USD', target=sym))
            path_list = [path for path in paths if len(path) > 6]
            for path in path_list:
                path.append('USD')
            glp_box.sym2path_list[sym] = path_list
    # find shortest path
    min_rtn = 999
    opt_path=[]
    for sym in glp_box.sym2path_list.keys():
        path_list = glp_box.sym2path_list[sym]
        rtn,op = computeMinAskRtn(path_list,ask_dict,bid_dict)
        if rtn < min_rtn:
            min_rtn = rtn
            opt_path = op
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

    ask_rtn = computePathAskRtn(opt_path,ask_dict,bid_dict)
    glp_box.trade_times[timestr] = ask_rtn
    print("ask rtn: {:.4e}".format(ask_rtn))
    glp_box.current_loop = opt_path
    glp_box.current_loop_rtn = ask_rtn
    syms, pos_type = generateTradeSyms(opt_path, 1,ask_dict.keys())
    glp_box.cur_syms = np.array(syms)

    price,lot = generatePriceLot(syms,pos_type,ask_dict,bid_dict)

    print("Trade info:")
    for sym,pos,pc,lz in zip(syms,pos_type,price,lot):
        print("{},{:2d},{:10.5f},{:.2f}".format(sym,pos,pc,lz))

    if not glpconf.isAllowPositions():
        return [],[],[],[]
    return syms, pos_type, price,lot
def find_sym_toclose(profit_list,nsyms,ncols):
    global glp_box
    sym_base = 200
    total_cap= sym_base*len(glp_box.cur_syms)
    isFullLoop = True if nsyms == len(glp_box.cur_syms) else False
    
    profit_arr = np.array(profit_list).reshape((nsyms,ncols))
    profit_arr = profit_arr+sym_base
    profits = np.transpose(profit_arr)
    for k in range(profits.shape[1]):
        y = profits[1:,k]
        for i in range(1+k,profits.shape[1]):
            x = profits[:-1,i]
            pv = checkDep(x,y)
            print("pval between {} and {}: {}".format(k,i,pv))
            if pv < 0.02:
                print("corr: {}".format(np.corrcoef(x,y)))


    if isFullLoop:
        profits = profits / total_cap
    else:
        print("TODO: ",len(glp_box.cur_syms))
    x = profits[:-1,:]
    y = profits[1:,:]
    model = sm.OLS(y,x).fit()
    # print(model.params)
    P = model.params
    P[P<0] = 0
    for i in range(P.shape[0]):
        P[i,:] = P[i,:]/sum(P[i,:])

    res = profits[-1,:]
    f0 = profits[0,:]
    # pdb.set_trace()
    for i in range(profits.shape[0]-1):
        res = np.matmul(res,P)
        f0  = np.matmul(f0,P)
    latest_profit = profits[-1,:]*total_cap-sym_base
    fitted_profit = f0*total_cap-sym_base
    pred_profit = res*total_cap-sym_base
    np.set_printoptions(floatmode='fixed', precision=2)
    print("latest profit: ",latest_profit)
    print("fitted profit: ", fitted_profit)
    print("pred_profit:   ",pred_profit)

    return np.argmin(pred_profit)

def get_loop():
    global glp_box
    return glp_box.current_loop
def get_loop_rtn():
    global glp_box
    return glp_box.current_loop_rtn
def compute_slope(list_x,list_y):
    x = np.array(list_x)
    y = np.array(list_y)
    p = np.polyfit(x,y,1)
    print("x: ",x)
    print("y: ", y)
    print("slope: ",p[0])
    return p[0]

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

    df = pd.DataFrame(list(glp_box.trade_times.items()), columns=['TRADE_TIME', 'MIN_ASK_RTN'])
    df.to_csv('trade_times.csv',index=False)
    print("trade times dumped to trade_times.csv")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} <mtn.yaml>'.format(sys.argv[0]))
        sys.exit(1)

    symlib = init(sys.argv[1])
    print("all trade pairs: ", symlib, len(symlib))

    # prices = np.random.random(28).tolist()
    ask = [0.8879, 0.58802, 93.268, 1.08173, 0.66902, 0.6628, 105.505, 159.17, 1.64205, 1.459, 0.96736, 0.85313, \
           153.505, 1.7812000000000001, 1.10401, 1.92441, 1.7127400000000002, 1.1325, 180.0, 2.08668, 1.29421, 0.81976, \
           0.54253, 86.361, 0.62068, 1.3218, 0.87535, 139.583]

    bid = ask
    syms, pos_type, price, lot = process_quote("2023-07-28 05:00:00", ask,bid)

    syms, pos_type, price, lot = process_quote("2023-07-28 05:00:00", ask, bid)

    loop = get_loop()
    print("current loop: ",loop)
    print("current loop rtn: ", get_loop_rtn())

    for s, p,pc,lz in zip(syms, pos_type,price,lot):
        print(s, p, pc,lz)

    profits = np.random.random((1,8*14*6))*500-250
    find_sym_toclose(profits.tolist(),8,14*6)


    finish()