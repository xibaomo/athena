#include "mb_pairtrader.h"
#include "pyrunner/pyrunner.h"
#include "basics/utils.h"
#include "histogram/histogram.h"
#include <fstream>
#include <numeric>
using namespace std;
using namespace athena;
Message
MinBarPairTrader::processMsg(Message& msg) {
    Message outmsg;
    FXAct action = (FXAct) msg.getAction();
    switch(action) {
    case FXAct::ASK_PAIR:
        outmsg = procMsg_ASK_PAIR(msg);
        break;
    case FXAct::PAIR_HIST_X:
        Log(LOG_INFO) << "X min bars arrive";
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            loadHistoryFromMsg(msg,m_minbarX,m_openX);
            Log(LOG_INFO) << "Min bar X loaded";
        });
        break;
    case FXAct::PAIR_HIST_Y: {
        outmsg = procMsg_PAIR_HIST_Y(msg);
    }
    break;
    case FXAct::PAIR_MIN_OPEN:
        outmsg = procMsg_PAIR_MIN_OPEN(msg);
        break;

    case FXAct::SYM_HIST_OPEN:
        outmsg = procMsg_SYM_HIST_OPEN(msg);
        break;
    case FXAct::PAIR_POS_PLACED:
        outmsg = procMsg_PAIR_POS_PLACED(msg);
        break;
    case FXAct::PAIR_POS_CLOSED:
        outmsg = procMsg_PAIR_POS_CLOSED(msg);
        break;
    default:
        break;
    }

    switch((FXAct)outmsg.getAction()) {
    case FXAct::PLACE_BUY:
        Log(LOG_INFO) << "Action: buy Y";
        break;
    case FXAct::PLACE_SELL:
        Log(LOG_INFO) << "Action: sell Y";
        break;
    case FXAct::NOACTION:
        Log(LOG_INFO) << "No action";
        break;
    default:
        break;

    }

    return outmsg;
}

Message
MinBarPairTrader::procMsg_PAIR_POS_CLOSED(Message& msg) {
    Message outmsg;

    String key = msg.getComment();

    if (m_pairTracker.find(key) == m_pairTracker.end()) {
        auto v = splitString(key,"/");
        key = v[1] + "/" + v[0];
    }
    if (m_pairTracker.find(key) == m_pairTracker.end()) {
        Log(LOG_ERROR) << "Cannot find the pair: " + key;
        return outmsg;
    }

    PairStatus& ps = m_pairTracker[key];
    real32* pc = (real32*)msg.getData();
    ps["profit"] = pc[0];

    return outmsg;
}

Message
MinBarPairTrader::procMsg_PAIR_POS_PLACED(Message& msg) {
    String key = msg.getComment();

    m_pairTracker[key] = m_currStatus;

    Message outmsg;
    return outmsg;
}
Message
MinBarPairTrader::procMsg_SYM_HIST_OPEN(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);
    String sym = pack.str_vec[0];
    Log(LOG_INFO) << "Received history: " + sym;

    if (m_sym2hist.find(sym) != m_sym2hist.end()) {
        Log(LOG_ERROR) << "Duplicated symbol received: " + sym;
    }

    m_sym2hist[sym] = std::move(pack.real32_vec);

    Message out;
    return out;
}
Message
MinBarPairTrader::procMsg_PAIR_HIST_Y(Message& msg) {
    Log(LOG_INFO) << "Y min bars arrive";

    loadHistoryFromMsg(msg,m_minbarY,m_openY);
    Log(LOG_INFO) << "Min bar Y loaded";


    if (m_minbarX.size() != m_minbarY.size()) {
        Log(LOG_FATAL) << "Inconsistent length of X & Y";
    }

    real64 corr = computePairCorr(m_openX,m_openY);
    Log(LOG_INFO) << "Correlation: " + to_string(corr);

    linearReg(m_openX.size()-5000);

    m_initSpreadMean = m_spreadMean;
    m_initSpreadStd  = m_spreadStd;

    test_coint(m_openX,m_openY);

    Message outmsg(msg.getAction(),sizeof(real32),0);
    real32* pm = (real32*) outmsg.getData();
    pm[0] = m_linregParam.c1;

    return outmsg;
}
void
MinBarPairTrader::loadHistoryFromMsg(Message& msg, std::vector<MinBar>& v, std::vector<real32>& openvec) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    int histLen = pack.int32_vec[0];
    if (histLen == 0) {
        Log(LOG_INFO) << "No min bars from mt5";
        return;
    }

    int bar_size = pack.int32_vec[1];
    real32* pm = &pack.real32_vec[0];
    int nbars = msg.getDataBytes()/sizeof(real32)/bar_size;
    if (nbars != histLen) {
        Log(LOG_FATAL) << "No. of min bars inconsistent";
    }

    for (int i = 0; i < nbars; i++) {
        v.emplace_back("unknown_time",pm[0],pm[1],pm[2],pm[3],pm[4]);
        openvec.push_back(v.back().open);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History min bars loaded: " + to_string(v.size());

}

Message
MinBarPairTrader::procMsg_ASK_PAIR(Message& msg) {
    //selectTopCorr();

    String s1 = m_cfg->getPairSymX();
    String s2 = m_cfg->getPairSymY();
    String st = s1 + ":" + s2;

    Message outmsg(FXAct::ASK_PAIR,sizeof(int),st.size());
    outmsg.setComment(st);

    return outmsg;
}

void
MinBarPairTrader::linearReg(int start) {
    int len = m_openX.size()-start;
    real64* x = new real64[len];
    real64* y = new real64[len];
    real64* spread = new real64[len];
    int k=0;
    for (int i=start; i< m_openX.size(); i++) {
        x[k] = m_openX[i];
        y[k] = m_openY[i];
        k++;
    }

    m_linregParam = linreg(x,y,len);

    real64 rms = sqrt(m_linregParam.chisq/len);
    Log(LOG_INFO) << "Linear regression done: c0 = " + to_string(m_linregParam.c0)
                  + ", c1 = " + to_string(m_linregParam.c1);

    real64 mean_y = gsl_stats_mean(y,1,len);
    real64 ss_tot=0.;
    real64 ss_res=0.;
    for (int i=0; i < len; i++) {
        spread[i] = y[i] - m_linregParam.c1*x[i];
        ss_tot += pow(y[i]-mean_y,2);
        real64 err = m_linregParam.c0 + m_linregParam.c1*x[i] - y[i];
        ss_res += pow(err,2);
    }
    real64 r2 = 1-ss_res/ss_tot;
    Log(LOG_INFO) << "R2 = " + to_string(r2);

    compDistr(spread,len,&m_spreadMean,&m_spreadStd);

    const char* env = getenv("NO_ADF_TEST");
    if (env && atoi(env)==1) {
        ;
    } else {
        vector<real64> spv(spread,spread+len);
        int lookback = m_cfg->getStationaryCheckLookback();
        real64 pv = testADF(spv,len-lookback,len);
        if (pv > m_maxStatPValue) m_maxStatPValue = pv;

        m_currStatus["statPV"] = pv;
        Log(LOG_INFO) << "p-value of non-stationarity of spread: " + to_string(pv);
    }

    m_currStatus["profit"] = 0;
    m_currStatus["c0"] = m_linregParam.c0;
    m_currStatus["c1"] = m_linregParam.c1;
    m_currStatus["rms"] = rms;
    m_currStatus["r2"] = r2;
    //dump spread
//    ofstream ofs("spread.csv");
//    ofs<<"x,y,spread"<<endl;
//    for (int i=0; i < len; i++) {
//        ofs << m_openX[i]<<","<<m_openY[i]<<","<<spread[i] << endl;
//    }
//    ofs.close();
//
//    Log(LOG_INFO) << "spread dumped to spread.csv";
    delete[] x;
    delete[] y;
    delete[] spread;
}

void
MinBarPairTrader::compDistr(real64* data, int len, real64* mean_out, real64* sd_out) {
    real64 minsp,maxsp;
    gsl_stats_minmax(&minsp,&maxsp,data,1,len);

    real64 mean = gsl_stats_mean(data,1,len);
    real64 sd   = gsl_stats_sd_m(data,1,len,mean);

    if(mean_out)    *mean_out=mean;
    if(sd_out)      *sd_out=sd;

    Log(LOG_INFO) << "Spread mean: " + to_string(mean) + ", std: " + to_string(sd);
    Log(LOG_INFO) << "Max deviation/std: " + to_string((minsp-mean)/sd) + ", "
                  + to_string((maxsp-mean)/sd);

    vector<real64> bins = generateBins(minsp,maxsp,mean,sd,1.);
    vector<real64> v_data(data,data+len);
    vector<size_t> hist = histogram(v_data,bins);

    unordered_map<String,size_t> bin2count;
    for(size_t i=0; i < hist.size(); i++) {
        bin2count[to_string(bins[i])] = hist[i];
    }

    size_t sum = std::accumulate(hist.begin(),hist.end(),0);
    int s=0;
    size_t sn=0;
    while(1) {
        real64 rs = mean+s*sd;
        real64 ls = mean-(s+1)*sd;
        if (rs> bins.back()) {
            Log(LOG_INFO) << to_string(s+1) + "std exceeds max";
            break;
        }
        if (ls < bins[0]) {
            Log(LOG_INFO) << to_string(s+1) + "std exceeds min";
            break;
        }
        String srs = to_string(rs);
        String sls = to_string(ls);
        if (bin2count.find(srs)==bin2count.end())
            Log(LOG_FATAL) << "Fail to find right std: " + srs;
        if (bin2count.find(sls)==bin2count.end())
            Log(LOG_FATAL) << "Fail to find left std: " + sls;

        sn += bin2count[srs] + bin2count[sls];
        real64 rat = (real64)sn/sum;
        String msg = "[-" + to_string(s+1) + "," + to_string(s+1) + "]std: " + to_string(rat);
        Log(LOG_INFO) << msg;

        s++;
    }
}

Message
MinBarPairTrader::procMsg_PAIR_MIN_OPEN(Message& msg) {
    Message outmsg(sizeof(real32),0);

    SerializePack pack;
    unserialize(msg.getComment(),pack);

    real32* pm = &pack.real32_vec[0];
    m_openX.push_back(pm[0]);
    m_openY.push_back(pm[1]);
    real64 x = pm[0];
    real64 y = pm[1];
    real32 y_pv = pm[2]; // point digits
    real32 y_pd = pm[3]; // point value in dollar

    //dumpVectors("open_price.csv",m_openX,m_openY);

    String timeStr = pack.str_vec[0];

    Log(LOG_INFO) <<"";

    Log(LOG_INFO) << "Mt5 time: " + timeStr + ", X: " + to_string(x)
                  + ", Y: " + to_string(y);

    real64 corr = computePairCorr(m_openX,m_openY);
    Log(LOG_INFO) << "Correlation so far: " + to_string(corr);
    if (fabs(corr) < m_cfg->getCorrBaseline()) {
        outmsg.setAction(FXAct::NOACTION);
        Log(LOG_ERROR) << "Correlation lower than threshold";
        return outmsg;
    }

//    if (!test_coint(m_openX,m_openY)) {
//        outmsg.setAction(FXAct::NOACTION);
//        return outmsg;
//    }

    linearReg(m_openX.size()-5000);

    m_currStatus["rms"] = m_currStatus["rms"] / y_pv*y_pd;

    Log(LOG_INFO) << "rms = $" + to_string(m_currStatus["rms"]) + " (per unit volume)";

    real64 spread = y - m_linregParam.c1*x;

    real64 thd = m_cfg->getLowThresholdStd();

    real64 dev_val = m_spreadStd/y_pv*y_pd;
    Log(LOG_INFO) << "std = $" + to_string(dev_val) + " (per unit volume)";

    real64 fac = (spread - m_spreadMean)/m_spreadStd;
    Log(LOG_INFO) << " ====== err/std: " + to_string(fac) + " ======";

    m_currStatus["err"] = fac;
    m_currStatus["spread"] = spread;
    real32* pcm = (real32*)outmsg.getData();
    pcm[0] = m_linregParam.c1;

    if ( fac > thd && fac < 3) {
        outmsg.setAction(FXAct::PLACE_SELL);
    } else if( fac < -thd && fac > -3) {
        outmsg.setAction(FXAct::PLACE_BUY);
    } else if ( fac * m_prevSpread < 0) {
        //outmsg.setAction(FXAct::CLOSE_ALL_POS);
        outmsg.setAction(FXAct::NOACTION);
    } else {
        outmsg.setAction(FXAct::NOACTION);
    }
    m_prevSpread = fac;

    if (m_currStatus["r2"] < m_cfg->getR2Baseline()) {
        outmsg.setAction(FXAct::NOACTION);
    }

    if (m_currStatus["statPV"] >= m_cfg->getStationaryPVLimit()) {
        outmsg.setAction(FXAct::CLOSE_ALL_POS);
    }
    return outmsg;
}

void
MinBarPairTrader::finish() {

    dumpStatus();

    Log(LOG_INFO) << "Stationary check lookback: " + to_string(m_cfg->getStationaryCheckLookback());
    Log(LOG_INFO) << "Max stationary p-value: " + to_string(m_maxStatPValue);
}

void
MinBarPairTrader::dumpStatus() {
    ofstream ofs("all_pos_pair.csv");
    int k=0;
    for (auto& it : m_pairTracker) {
        auto& status = it.second;
        if (status.empty()) {
            continue;
        }
        if (status["profit"] == 0)
            continue;
        if (k==0) {
            // dump headers
            ofs << "tickets,";
            for(auto& itt : status) {
                ofs<<itt.first<<",";
            }
            ofs << "label\n";
            k++;
        }
        // value
        ofs << it.first<<",";
        for(auto& itt : status) {
            ofs << itt.second<<",";
        }
        if (status["profit"] > 0) {
            ofs << "1\n";
        }  else {
            ofs << "0\n";
        }

    }
    ofs.close();
}



