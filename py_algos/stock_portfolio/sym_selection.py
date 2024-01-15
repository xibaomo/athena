import pdb

from mkv_absorb import *
import numpy as np
import pdb

def corr2distScore(cm):
    nsyms = cm.shape[0]
    # pdb.set_trace()
    transmat = np.zeros((nsyms,nsyms))
    for i in range(nsyms):
        for j in range(i+1,nsyms):
            if cm[i,j] < 0:
                transmat[i,j] = -cm[i,j]
                transmat[j,i] = transmat[i,j]

    # pdb.set_trace()
    for i in range(nsyms):
        s = sum(transmat[i,:])
        if s==0:
            print("===================== loner found ==================")
        transmat[i,:] = transmat[i,:]/s

    # pdb.set_trace()
    I = np.eye(nsyms)
    ONE = np.ones((nsyms,nsyms))
    row_one = np.ones(nsyms)
    tmp = I - transmat + ONE
    tmp = np.linalg.inv(tmp)
    sol = np.matmul(row_one,tmp)
    return sol

def select_syms_corr_dist(df,num_syms):
    cm = df.corr().values
    s1 = corr2distScore(cm)
    cm2 = df.iloc[-30:,:].corr().values
    s2 = corr2distScore(cm2)
    score = np.array(s1+s2)
    # pdb.set_trace()
    sorted_id = np.argsort(score)[::-1]
    all_syms = df.keys().values[sorted_id]

    print(score[sorted_id])
    print(score.sum())

    return all_syms[:num_syms].tolist()