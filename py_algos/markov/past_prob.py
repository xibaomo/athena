import pandas as pd
import os,sys,pdb
import matplotlib.pyplot as plt
from markov import *
if __name__ == "__main__":
    csvfile = sys.argv[1]

    tar_time = sys.argv[2] + ' ' + sys.argv[3]

    odf = pd.read_csv(csvfile, sep='\t')

    ts = pd.to_datetime(odf['<DATE>'] + " " + odf['<TIME>'])

    tt = pd.to_datetime(tar_time)

    tarid = ts.index[ts == tt].tolist()[0]

    mkvcal = FirstHitProbCal(odf,501)
    lookback = 60*24

    ftu = int(sys.argv[-1])
    ftu = min(len(odf) - tarid, ftu)
    op = odf['<OPEN>'].values
    probs = []
    ps=[]
    rs = []
    kh=-1
    for i in range(10,ftu):
        tm = ts[tarid+i]
        if tm.second > 0 or tm.minute > 0:
            continue
        kh +=1
        hist_end=tarid+i
        hist_start = hist_end-lookback
        # lookback = hist_end-hist_start
        rtn = op[hist_end]/op[hist_start] - 1
        ps.append(op[hist_end])
        prtn = 0
        nrtn = 0
        if rtn==0:
            continue
        if rtn > 0:
            prtn = rtn
            nrtn = -prtn
            prob,_ = mkvcal.comp1stHitProb(hist_start,hist_end,prtn,nrtn,lookback)
        else:
            prtn = -rtn
            nrtn = rtn
            _,prob = mkvcal.comp1stHitProb(hist_start, hist_end, prtn, nrtn,lookback)
            prob=-prob
        print(ts[hist_end],prob)
        print ("rtn: ",rtn)
        probs.append(prob)
        rs.append(rtn)


    ps = np.array(ps)
    ps = ps/ps[0]-1
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(probs, 'b.-')
    ax1.plot(ps, 'r.-')

    plt.show()
