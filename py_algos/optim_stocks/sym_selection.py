import pdb
from mkv_absorb import *
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import math
from scipy.stats import spearmanr,shapiro
from scipy.stats import ks_2samp, anderson_ksamp, mannwhitneyu
import scipy.stats as stats

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
def transmat2dist(transmat,timesteps=100,exclude_isolation=True):
    nsyms = transmat.shape[0]
    x = np.ones([nsyms, 1])
    # pdb.set_trace()
    if exclude_isolation:
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

    print(sorted_syms[:num_selected_syms])
    print(score[sorted_id][:num_selected_syms])

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
        # pdb.set_trace()
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
    cal=MkvAbsorbCal(scoreconf.getMarkovSpeedPartitions(),scoreconf.getCDFType())
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
    # speeds = np.zeros(len(arg_list))
    # for i in range(len(arg_list)):
    #     print(i)
    #     if i==232:
    #         pdb.set_trace()
    #     speeds[i] = comp_mkv_speed(arg_list[i])

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

def select_syms_net_buy_power(df_close,df_vol,lookback):
    zero_cols = df_vol.columns[(df_vol==0).any()]
    df_vol = df_vol.drop(zero_cols,axis=1)
    df_close=df_close.drop(zero_cols,axis=1)
    nsyms = len(df_close.keys())
    daily_rtn = df_close.pct_change()
    df_buypower = daily_rtn/df_vol
    ma_df_buypower = df_buypower.rolling(window=lookback).mean()
    ma_df_buypower = ma_df_buypower.dropna()
    mu = ma_df_buypower.mean()
    sd = ma_df_buypower.std()
    # pdb.set_trace()
    scores = np.zeros(nsyms)
    for i in range(nsyms):
        pwr = ma_df_buypower.iloc[-1,i]
        th = mu.iloc[i] + sd.iloc[i]*2
        rng = 4*sd.iloc[i]
        if pwr >= th:
            scores[i] = 0
        else:
            scores[i] = abs(pwr-th)/rng

    # sorted_id = np.argsort(scores)[::-1]
    # scores = scores[sorted_id]
    # sorted_syms = df_close.keys()[sorted_id]
    #
    # for i in range(20):
    #     print("{}: {}".format(sorted_syms[i],scores[i]))
    # scores = standardize(scores)

    syms = select_syms_by_score(scores,df_close.keys(),0,20)
    # syms=[]
    return syms

def corr_lookback(lookback,df_sh,df_close):
    x = df_sh.rolling(window=int(lookback)).mean().values
    # pdb.set_trace()
    idx = ~np.isnan(x)
    x = x[idx]
    y = df_close.values[idx]
    # cr =  np.corrcoef(x,y)[0,1]
    cr,pv = spearmanr(x,y)
    # print("curretn lookback: ", lookback,cr)
    return -cr
def optimize_lookback(df_sh,df_close):
    b = 500
    a = 10
    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
    (a, b) = (min(a, b), max(a, b))
    h = b - a

    c = a + invphi2 * h
    d = a + invphi * h
    yc = corr_lookback(c,df_sh,df_close)
    yd = corr_lookback(d,df_sh,df_close)

    while b-a>1:
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = corr_lookback(c,df_sh,df_close)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = corr_lookback(d,df_sh,df_close)

    # print("best lookback: ",b,-corr_lookback(int(b),df_sh,df_close))
    best_cr = -corr_lookback(int(b),df_sh,df_close)
    return int(b), best_cr

def score_netbuypower_slope_corr_individual(df_sh,df_close):
    # pdb.set_trace()
    LOW=0
    lookback,cr = optimize_lookback(df_sh,df_close)
    if cr < 0.5:
        return LOW
    ma_sh = df_sh.rolling(window=lookback).mean()
    N = 31
    x = np.array([i for i in range(N)])
    y = ma_sh.values[-N:]
    p = np.polyfit(x,y,2)
    fy = np.polyval(p,x)
    if np.isnan(fy[-1]):
        return LOW
    if fy[-1] <= 0:
        return LOW
    df = fy[-1]-fy[-2]
    if df < 0:
        return LOW
    if np.isnan(df*cr):
        pdb.set_trace()
    return cr*fy[-1]*df

def score_netbuypower_slope_corr(df_sh,df_close):
    nsyms = len(df_sh.keys())
    scores = np.zeros(nsyms)
    # pdb.set_trace()
    for i in range(nsyms):
        scores[i] = score_netbuypower_slope_corr_individual(df_sh.iloc[:,i],df_close.iloc[:,i])
        # print(df_sh.keys()[i],scores[i])

    print("Raw scores:")
    for i in range(nsyms):
        if np.isnan(scores[i]) or scores[i] > 0:
            print(df_sh.keys()[i],scores[i])
    return scores

def score_buypower_mkvspeed_individual(arr):
    arr = arr[~np.isnan(arr)]
    mkvcal = MkvAbsorbCal(100,'gauss')
    mx = np.max(arr)*10
    p,sp = mkvcal.compUpAveSteps(arr,-mx,mx)

    return mx/sp

def score_buypower_mkv_speed(df_close,df_volval):
    pool = multiprocessing.Pool(processes=NUM_PROCS)
    df_rtn = df_close.pct_change()
    df_sh = df_rtn/df_volval

    args = [df_sh[key].values for key in df_sh.keys()]
    scores = pool.map(score_buypower_mkvspeed_individual,args)
    scores = np.array(scores)
    # pdb.set_trace()

    scores = standardize(scores)
    return scores

def score_price_mkvspeed_individual(arr):
    mx=.5
    arr = arr[~np.isnan(arr)]
    mkvcal = MkvAbsorbCal(100,'laplace')
    sp = mkvcal.compUpAveSteps(arr,-mx,mx)

    return mx/sp

def score_price_mkv_speed(df_close,df_volval):
    pool = multiprocessing.Pool(processes=NUM_PROCS)
    df_rtn = df_close.pct_change()

    args = [df_rtn[key].values for key in df_rtn.keys()]
    scores = pool.map(score_price_mkvspeed_individual,args)
    # scores=[]
    # for i in range(len(args)):
    #     s = score_price_mkvspeed_individual(args[i])
    #     scores.append(s)
    scores = np.array(scores)
    # pdb.set_trace()

    scores = standardize(scores)
    return scores

def score_return_flow(df_close,risk_free_rtn):
    df_rtn = df_close.pct_change()
    df_rtn = df_rtn.dropna()
    # df_rtn['risk_free'] = risk_free_rtn/250
    sym_rtn = df_rtn.mean().values
    sym_std = df_rtn.std().values
    metrics = sym_rtn/sym_std

    nsyms = df_rtn.shape[1]
    transmat = np.zeros((nsyms,nsyms))
    for i in range(nsyms):
        for j in range(nsyms):
            if i == j:
                continue
            if metrics[j] <= 0:
                continue
            if metrics[j] <= metrics[i]:
                continue

            transmat[i,j] = metrics[j] - metrics[i]

    for i in range(nsyms):
        s = np.sum(transmat[i,:])
        if s == 0:
            transmat[i,i] = 1
        else:
            transmat[i,:] = transmat[i,:]/s

    score = transmat2dist(transmat,timesteps=3,exclude_isolation=False)
    return score


def cost_return_per_risk(args,disp_result=False):  # args: [(sym1,rtns1),(sym2,rtns),...]
    nsyms = len(args)
    # if nsyms < 2:
    #     return 0,np.array([])
    len_hist = len(args[0][1])
    port_rtns = np.zeros(len_hist)
    for i in range(len_hist):
        rtns = [args[j][1][i] for j in range(nsyms)]
        # pdb.set_trace()
        port_rtns[i] = np.mean(rtns)
    mu = np.mean(port_rtns)
    sd = np.std(port_rtns)
    return -mu / sd, port_rtns

def series_stat_cost(series,segment_size=60):
    num_segments = len(series) // segment_size
    ks_results = []
    for i in range(num_segments - 1):
        segment1 = series[i * segment_size:(i + 1) * segment_size]
        segment2 = series[(i + 1) * segment_size:(i + 2) * segment_size]
        if (i + 2) * segment_size > len(series):
            segment2 = series[(i + 1) * segment_size:]
        ks_stat, p_value = ks_2samp(segment1, segment2)
        ks_results.append(p_value)
    return np.mean(ks_results)

def cost_mkv_speed(args,partitions=100,lb_rtn=-.15, ub_rtn=.15,stationary_days = 40,up_prob_wt=5,cdf_type='emp',disp_result=False):
    # tic = time.time()
    nsyms = len(args)
    len_hist = len(args[0][1])
    port_rtns = np.zeros(len_hist)
    for i in range(len_hist):
        rtns = [args[j][1][i] for j in range(nsyms)]
        # pdb.set_trace()
        port_rtns[i] = np.mean(rtns)

    mkvcal = MkvAbsorbCal(partitions,cdf_type)
    p,sp = mkvcal.compWinProb(port_rtns,lb_rtn,ub_rtn)
    # print("mkv takes: ", time.time()-tic)

    mid = -stationary_days
    res = ks_2samp(port_rtns[:mid], port_rtns[mid:])
    timecost = sp/60.
    if timecost < 0:
        timecost=0
    stat_cost = series_stat_cost(port_rtns)
    if disp_result:
        print("Up prob: {:.3f}, ave steps: {}".format(p,int(sp)))
        print("stationary cost: ", stat_cost)
    cost = -p*up_prob_wt + timecost*0.5 - stat_cost
    return cost,port_rtns
def cost_stationary(args,disp_result=False):
    nsyms = len(args)
    len_hist = len(args[0][1])
    port_rtns = np.zeros(len_hist)
    for i in range(len_hist):
        rtns = [args[j][1][i] for j in range(nsyms)]
        # pdb.set_trace()
        port_rtns[i] = np.mean(rtns)

    mid = -40
    res = ks_2samp(port_rtns[:mid], port_rtns[mid:])
    if disp_result:
        print(res)
    return -res[1],port_rtns

def rtn_per_risk(rtns):
    mu = np.mean(rtns)
    sd = np.std(rtns)
    return mu/sd

def check_log_normal(arr):
    data = np.log(arr)
    shapiro_test = shapiro(data)
    return shapiro_test
def check_log_laplace(arr):
    log_data = np.log(arr)
    loc, scale = stats.laplace.fit(log_data)
    ks_test = stats.kstest(log_data, 'laplace', args=(loc, scale))
    return ks_test
