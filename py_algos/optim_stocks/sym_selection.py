import pdb
from mkv_absorb import *
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

NUM_PROCS = multiprocessing.cpu_count()-2

def standardize(score):
    # normalize score to range (0,1)
    score = score - np.min(score)
    score = score / np.max(score)
    return score
def __transmat2dist(transmat):
    nsyms = transmat.shape[0]
    I = np.eye(nsyms)
    ONE = np.ones((nsyms, nsyms))
    row_one = np.ones(nsyms)
    tmp = I - transmat + ONE
    tmp = np.linalg.inv(tmp)
    sol = np.matmul(row_one, tmp)
    return sol
def transmat2dist(transmat,timesteps=100):
    nsyms = transmat.shape[0]
    x = np.ones([nsyms, 1])
    # pdb.set_trace()
    for i in range(nsyms):
        if transmat[i, i] == 1:
            x[i, 0] = 0

    t_trans = transmat.transpose()
    for i in range(timesteps):
        x = np.matmul(t_trans, x)
    return x.flatten()
def corr2distScore(cm, df, timesteps):
    nsyms = cm.shape[0]
    # pdb.set_trace()
    transmat = np.zeros((nsyms, nsyms))
    for i in range(nsyms):
        for j in range(i+1, nsyms):
            if cm[i, j] <= -.05:
                # x = df.iloc[:, i].values.reshape([-1, 1])
                # y = df.iloc[:, j].values
                # transmat[i, j] = mutual_info_regression(x, y)
                transmat[i, j] = -cm[i, j]
                transmat[j, i] = transmat[i, j]

    # pdb.set_trace()
    for i in range(nsyms):
        s = sum(transmat[i, :])
        if s == 0:
            # print("===================== loner found ==================")
            transmat[i, i] = 1
        else:
            transmat[i, :] = transmat[i, :]/s

    # pdb.set_trace()
    score = transmat2dist(transmat,timesteps)
    return score
def check_mutual_info(df, cm):
    c=[]
    s=[]
    for i in range(cm.shape[0]):
        for j in range(i+1, cm.shape[1]):
            if cm[i, j] < 0:
                x = df.iloc[:, i].values.reshape([-1, 1])
                y = df.iloc[:, j].values
                score = mutual_info_regression(x, y)[0]
                c.append(cm[i, j])
                s.append(score)
                print("corr: {}, mutual_score: {}".format(cm[i, j], score))
                if cm[i, j] > -0.1 and score > 1.0:
                    plt.plot(x, y, '.')
                    plt.show()

    plt.plot(c, s, '.')
    plt.show()

def select_syms_corr_dist(df, num_syms, short_wt=1.2, timesteps=100, random_select=False):
    print("hisotry length: ", len(df))
    df = removeCollapsedSym(df)
    cm = df.corr().values
    # check_mutual_info(df, cm)
    s1 = corr2distScore(cm, df,timesteps)
    df2 = df.iloc[-30:, :]
    cm2 = df2.corr().values
    s2 = corr2distScore(cm2, df2,timesteps)
    score = np.array(s1+s2*short_wt)
    # pdb.set_trace()
    sorted_id = np.argsort(score)[::-1]
    all_syms = df.keys().values[sorted_id]

    print(score[sorted_id])
    print(score.sum())

    if random_select:
        print("Randomly pick among top {}".format(num_syms*2))
        candidates = all_syms[:num_syms*2]
        np.random.shuffle(candidates)
        return candidates[:num_syms].tolist()
    return all_syms[:num_syms].tolist()

def normalizeTransmat(transmat):
    for i in range(transmat.shape[0]):
        s = sum(transmat[i, :])
        if s == 0:
            # print("===================== loner found ==================")
            transmat[i, i] = 1
        else:
            transmat[i, :] = transmat[i, :]/s
    return transmat
def cal_slope(pair):
    # pdb.set_trace()
    x,y=pair
    return np.polyfit(x,y,1)[0]
def computeSlopeTransmat(df):
    pool = multiprocessing.Pool(processes=NUM_PROCS)

    nsyms = len(df.keys())
    transmat = np.zeros((nsyms, nsyms))
    for i in range(nsyms):
        x = df[df.keys()[i]].values
        pairs = [(x,df[s].values) for s in df.keys()]
        slopes = pool.map(cal_slope,pairs)
        for j in range(nsyms):
            if slopes[j] >=0:
                continue
            transmat[i,j] = -slopes[j]
        # for j in range(nsyms):
        #     if i==j:
        #         continue
        #     y = df[df.keys()[j]]
        #     slope = np.polyfit(x, y, 1)[0]
        #     if slope >= 0:
        #         continue
        #     transmat[i, j] = -slope

    transmat = normalizeTransmat(transmat)
    return transmat

def removeCollapsedSym(df,thd=.5):
    # pdb.set_trace()
    return df
    cols = []
    for i in range(len(df.keys())):
        init_price = df.iloc[0,i]
        end_price = df.iloc[-1,i]
        if end_price/init_price <= thd:
            cols.append(df.keys()[i])
    df = df.drop(cols,axis=1)
    print("removed syms: ", cols)
    return df
def select_syms_by_score(score,all_syms,random_select,num_selected_syms):
    # pdb.set_trace()
    sorted_id = np.argsort(score)[::-1]
    sorted_syms = all_syms[sorted_id]

    print(score[sorted_id])
    print(score.sum())

    if random_select:
        print("Randomly pick among top {}".format(num_selected_syms * 2))
        candidates = sorted_syms[:num_selected_syms * 2].tolist()
        np.random.shuffle(candidates)
        return candidates[:num_selected_syms]
    return sorted_syms[:num_selected_syms].tolist()

def select_syms_slope_dist(df, num_syms, short_wt, timesteps, random_select):
    df = removeCollapsedSym(df)
    print("hisotry length: ", len(df))
    transmat1 = computeSlopeTransmat(df)
    s1 = transmat2dist(transmat1,timesteps)

    df2 = df.iloc[-30:,:]
    transmat2 = computeSlopeTransmat(df2)
    s2 = transmat2dist(transmat2,timesteps)

    score = np.array(s1 + s2 * short_wt)
    score = score - np.min(score)
    score = score/np.max(score)

    # score of stock individual performance
    score_idv1 = score_by_rtn_per_risk(df)
    score_idv2 = score_by_rtn_per_risk(df2)
    score_idv = score_idv1 + score_idv2*short_wt
    score_idv = score_idv - np.min(score_idv)
    score_idv = score_idv/np.max(score_idv)

    selected_syms = select_syms_by_score(score+score_idv,df.keys(),random_select,num_syms)

    return selected_syms
def computeSlopeCorrTransmat(df):
    pool = multiprocessing.Pool(processes=NUM_PROCS)

    cm = df.corr().values
    nsyms = len(df.keys())
    transmat = np.zeros((nsyms, nsyms))
    for i in range(nsyms):
        x = df[df.keys()[i]].values
        pairs = [(x,df[s].values) for s in df.keys()]
        slopes = pool.map(cal_slope,pairs)
        for j in range(nsyms):
            if slopes[j] >=0 or cm[i,j] >=0:
                continue
            transmat[i,j] = slopes[j]*cm[i,j]
        # for j in range(nsyms):
        #     if i==j:
        #         continue
        #     y = df[df.keys()[j]]
        #     slope = np.polyfit(x, y, 1)[0]
        #     if slope >= 0:
        #         continue
        #     transmat[i, j] = -slope

    transmat = normalizeTransmat(transmat)
    return transmat
def select_syms_corr_slope_dist(df, num_syms, short_wt, timesteps, random_select):
    print("hisotry length: ", len(df))
    transmat1 = computeSlopeCorrTransmat(df)
    s1 = transmat2dist(transmat1, timesteps)

    df2 = df.iloc[-30:, :]
    transmat2 = computeSlopeCorrTransmat(df2)
    s2 = transmat2dist(transmat2, timesteps)

    score = np.array(s1 + s2 * short_wt)
    score = score - np.min(score)
    score = score / np.max(score)

    selected_syms = select_syms_by_score(score, df.keys(), random_select, num_syms)

    return selected_syms
def score_by_rtn_per_risk(df):
    daily_rtn = df.pct_change()
    mu = daily_rtn.mean().values
    sd = daily_rtn.std().values
    score = mu/sd
    # score = score/np.mean(score)
    return score

def score_corr_slope_dist(df,timesteps,short_wt):
    transmat1 = computeSlopeCorrTransmat(df)
    s1 = transmat2dist(transmat1, timesteps)

    df2 = df.iloc[-30:, :]
    transmat2 = computeSlopeCorrTransmat(df2)
    s2 = transmat2dist(transmat2, timesteps)

    score = np.array(s1 + s2 * short_wt)
    score = score - np.min(score)
    score = score / np.max(score)

    return score

def score_volval_mean_offset(df_vv,short_wt):
    vv1 = df_vv.sum().values
    s1 = vv1 - np.mean(vv1)

    vv2 = df_vv.iloc[-30:,:].sum()
    s2 = vv2 - np.mean(vv2)

    score = s2
    score = score - np.min(score)
    score = score/np.max(score)

    return score

def comp_mkv_speed(args):
    scoreconf,rtn_arr = args
    cal=MkvAbsorbCal(scoreconf.getMarkovSpeedPartitions())
    bnd = scoreconf.getMarkovSpeedBound()
    p,sp = cal.compWinProb(rtn_arr,-bnd,bnd)
    if p < .5:
        sp=-sp
    return bnd/sp
def score_mkv_speed(dff,scoreconf):
    lookback = scoreconf.getMarkovSpeedLookback()
    df = dff.iloc[-lookback:,:]
    daily_rtn = df.pct_change()
    arg_list = [(scoreconf,daily_rtn[sym].values) for sym in df.keys()]

    pool = multiprocessing.Pool(processes=NUM_PROCS)
    speeds = pool.map(comp_mkv_speed,arg_list)

    score = np.array(speeds)
    score = standardize(score)
    return score

def score_specific_heat(df_typ,df_volval,scoreconf):
    lookback = scoreconf.getSpecificHeatLookback()
    df_typ = df_typ.iloc[-lookback*5:,:]
    df_volval=df_volval.iloc[-lookback*5:,:]
    df_sh = df_typ.copy()
    for i in range(len(df_typ)):
        for j in range(len(df_typ.keys())):
            if i==0:
                continue
            dp = df_typ.iloc[i,j] - df_typ.iloc[i-1,j]
            if df_volval.iloc[i,j]==0:
                df_sh.iloc[i,j] = 0     #assume buy power is balanced with sell
            else:
                df_sh.iloc[i,j] = dp/(df_volval.iloc[i,j])  #avoid divergence

    scores=np.zeros(len(df_typ.keys()))
    crs=np.zeros(len(scores))
    for j in range(len(df_typ.keys())):
        # pdb.set_trace()
        sh = df_sh.iloc[:,j].rolling(window=lookback).mean().values
        tp = df_typ.iloc[:,j].values
        cr = np.corrcoef(sh[lookback:-1],tp[lookback+1:])[0,1]
        if cr < 0:
            cr=0
        # print(j)
        scores[j] = sh[-1]
        crs[j] = cr
    scores = scores *crs
    scores = standardize(scores)
    return scores

def score_money_flow(df_close,df_volval,scoreconf):
    lookback=90
    sid = len(df_close)-lookback - 1
    scores = np.zeros(len(df_close.keys()))
    for j in range(len(df_close.keys())):
        # pdb.set_trace()
        money_in=0
        money_out=0
        for i in range(sid,len(df_close)):
            if df_close.iloc[i,j] >=df_close.iloc[i-1,j]:
                money_in = money_in + df_volval.iloc[i,j]
            else:
                money_out = money_out + df_volval.iloc[i,j]
        scores[j] = money_in/money_out

    scores = standardize(scores)
    return scores

def dallorvol2transmat(df_typ, df_volval_org):
    df_volval = df_volval_org.copy()
    cm = df_typ.corr().values
    nsyms = len(df_typ.keys())
    for i in range(len(df_typ)):
        for j in range(len(df_typ.keys())):
            if i==0:
                continue
            if df_typ.iloc[i,j] < df_typ.iloc[i-1,j]:
                v = df_volval.iloc[i,j]
                df_volval.iloc[i,j] = -v
    transmat = np.zeros([nsyms,nsyms])
    for row in range(1,len(df_volval)):
        tmp = np.zeros([nsyms,nsyms])
        for i in range(nsyms):
            for j in range(i+1,nsyms):
                r = df_volval.iloc[row,j]/df_volval.iloc[row,i]*cm[i,j]
                if r > 0 and cm[i,j] < 0:
                    tmp[i,j] = r
                    tmp[j,i] = 1./r
        transmat = transmat + tmp
        print('transmat done at row ',row)
    for i in range(nsyms):
        s = sum(transmat[i,:])
        transmat[i,:] = transmat[i,:]/s
    return transmat

def score_dollarvol_dist(df_typ,df_volval):
    zero_cols = df_volval.columns[(df_volval==0).any()]
    print("zero cols: ", zero_cols)
    df_typ = df_typ.drop(zero_cols,axis=1)
    df_volval = df_volval.drop(zero_cols,axis=1)
    transmat = dallorvol2transmat(df_typ,df_volval)
    s = transmat2dist((transmat))

    s = standardize(s)
    return s

def cal_linear_offset(x,y):
    coef = np.polyfit(x,y,1)
    rem = y - (coef[0]*x + coef[1])
    sigma = np.std(rem)
    return rem[-1]/sigma
def score_by_pair_slope(df_close):
    cm = df_close.corr().values
    syms = df_close.keys()
    sym2score = {}
    for s in syms:
        sym2score[s] = 0

    lw = 0
    for i in range(cm.shape[0]):
        for j in range(i+1,cm.shape[1]):
            if cm[i,j] > .85:
                x = df_close[syms[i]].values
                y = df_close[syms[j]].values
                s = cal_linear_offset(x,y)
                sym2score[syms[j]] += -s
                if -s < 0 and sym2score[syms[j]] < lw:
                    lw = sym2score[syms[j]]
                s = cal_linear_offset(y,x)
                sym2score[syms[i]] += -s
                if -s < 0 and sym2score[syms[i]] < lw:
                    lw = sym2score[syms[i]]

    # sym2score = {key: value for key, value in sym2score.items() if value != 0}
    for s in sym2score.keys():
        if sym2score[s] == 0:
            sym2score[s] = lw

    scores = list(sym2score.values())
    scores = standardize(scores)
    return scores



