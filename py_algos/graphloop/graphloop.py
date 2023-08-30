import pdb,copy
import pandas as pd
import sys,os
import networkx as nx
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

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
def createGraph(nodes,all_syms):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in nodes:
        G.add_node(node)

    for col in all_syms:
        # pdb.set_trace()
        s1 = col[:3]
        s2 = col[3:]

        G.add_edge(s1,s2,weight=1)
        G.add_edge(s2,s1,weight=1)

    return G
def createGraphWeight(nodes,all_syms,mid_dict,base=0.):
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in nodes:
        G.add_node(node)
    for col in all_syms:
        # pdb.set_trace()
        s1 = col[:3]
        s2 = col[3:]

        w = np.log(mid_dict[col])
        G.add_edge(s1, s2, weight=w-base+.1)
        G.add_edge(s2, s1, weight=-w-base+.1)

    return G

def computePathAskRtn(path, ask_dict,bid_dict):
    rtn = 0.
    for i in range(len(path)-1):
        src = path[i]
        dst = path[i+1]
        sym = src+dst
        w=0.
        if sym in ask_dict.keys():
            w = np.log(ask_dict[sym])
        else:
            sym = dst+src
            if not sym in bid_dict.keys():
                # pdb.set_trace()
                print("ERROR!!!!!!!!!!!!!! sym not found:",sym)
                sys.exit(1)
            w = -np.log(bid_dict[sym])
        rtn+=w

    return rtn

def computePathBidRtn(path, ask_dict,bid_dict):
    rtn = 0.
    for i in range(len(path)-1):
        src = path[i]
        dst = path[i+1]
        sym = src+dst
        w=0.
        if sym in bid_dict.keys():
            w = np.log(bid_dict[sym])
        else:
            sym = dst+src
            if not sym in ask_dict.keys():
                print("ERROR!!!!!!!!!!!!!! sym not found:",sym)
                sys.exit(1)
            w = -np.log(ask_dict[sym])
        rtn+=w

    return rtn

def computePathMidRtn(path,mid_dict):
    rtn = 0.
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        sym = src + dst
        w = 0.
        if sym in mid_dict.keys():
            w = np.log(mid_dict[sym])
        else:
            sym = dst + src
            if not sym in mid_dict.keys():
                # pdb.set_trace()
                print("ERROR!!!!!!!!!!!!!! sym not found:", sym)
                sys.exit(1)
            w = -np.log(mid_dict[sym])
        rtn += w
    return rtn
def computeMinMidRtn(path_list,ask_dict,bid_dict):
    mid_dict = {}
    for key in ask_dict.keys():
        p = (ask_dict[key] + bid_dict[key])*.5
        mid_dict[key] = p
    min_rtn = 999
    opt_path = path_list[0]
    for path in path_list:
        rtn = computePathMidRtn(path, mid_dict)
        if rtn < min_rtn:
            min_rtn = rtn
            opt_path = path
    return min_rtn, opt_path
def computeMinAskRtn(path_list,ask_dict,bid_dict):
    min_rtn = 999
    opt_path = path_list[0]
    for path in path_list:
        rtn = computePathAskRtn(path,ask_dict,bid_dict)
        if rtn < min_rtn:
            min_rtn = rtn
            opt_path = path
    return min_rtn,opt_path

def computeMaxBidRtn(path_list,ask_dict,bid_dict):
    max_rtn = -999
    opt_path = path_list[0]
    for path in path_list:
        rtn = computePathBidRtn(path,ask_dict,bid_dict)
        if rtn > max_rtn:
            max_rtn = rtn
            opt_path = path
    return max_rtn,opt_path
def __computeLimitRtns(G,path_list, src_node,tar_node,prices):
    all_paths = copy.deepcopy(path_list)

    # print("list of all paths created")
    pair = tar_node + src_node
    w=0.
    if not pair in prices.keys():
        pair = src_node + tar_node
        w = -np.log(prices[pair])
    else:
        w = np.log(prices[pair])

    wts = []
    high_score=-1.
    high_path = None
    low_score = 1
    low_path = None
    for path in all_paths:
        total_weight = sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)) + w
        score = total_weight
        # score = total_weight / len(path)
        if score > high_score:
            high_score = score
            high_path = path
        if score < low_score:
            low_score = score
            low_path = path

    high_path.append(src_node)
    low_path.append(src_node)
    return high_score,high_path,low_score,low_path

def findTradePairPosRtn(df,G,path):
    hw = -1
    hw_edge =[]
    # find highest weight
    for i in range(len(path)-1):
        w = G[path[i]][path[i + 1]]["weight"]
        if w > hw:
            hw = w
            hw_edge = [path[i], path[i+1]]
    sym,isflip = getTruePair(hw_edge,df.keys())
    pos_type = -1  # 1 - long, -1 - short
    if isflip:
        pos_type = 1
    return sym,pos_type

def getTruePair(edge,all_pairs):
    isflip = False
    sym = edge[0] + edge[1]
    if not sym in all_pairs:
        sym = edge[1] + edge[0]
        isflip = True
    return sym,isflip

def plot_double_y(y1,y2):
    fig, ax1 = plt.subplots()
    ax1.plot(y1,'*-')

    # Create a second y-axis (right)
    ax2 = ax1.twinx()
    ax2.plot(y2,'r.-')

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
    data = yf.download(syms.tolist(), start = start_date, end = end_date,interval='5m')['Open']
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
    # prices = df.iloc[3,:]
    # G = createGraph(nodes,prices)
    #
    # for sym in nodes:
    #     if sym == 'USD':
    #         continue
    #     mi,mx,_ = computeLimitRtns(G,'USD',sym)
    #     print("USD->{}: min: {:.4e}, max: {:.4e}".format(sym,mi,mx))

    tar_node = 'EUR'
    scores = []
    tar_prices=[]
    fwd = 5
    profits=[]
    for i in range(len(df)-fwd):
        prices = df.iloc[i,:]
        G = createGraph(nodes,prices)
        try:
            high_rtn,high_path,low_rtn,low_path= computeLimitRtns(G,'USD',tar_node,prices)
            scores.append(high_rtn)
            # only look at high for now
            if high_rtn > 3e-3:
                sym,pos_type = findTradePairPosRtn(df,G,high_path)

                sym_rtn = df[sym][i+fwd]/df[sym][i] - 1
                L = len(high_path)-1
                print("tid: {}, length: {},  profit: {:.2f}".format(i, len(high_path),sym, pos_type,\
                                                                                        high_rtn*1e5))
                # p = sym_rtn*1e5*pos_type
                # if not np.isnan(p):
                #     profits.append(p)

        except:
            continue
    print ("total profit: {:.2f}".format(sum(profits)))

    plt.plot(scores,'*-')
    plt.show()


