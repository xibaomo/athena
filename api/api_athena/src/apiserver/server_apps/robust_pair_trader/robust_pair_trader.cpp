/*
 * =====================================================================================
 *
 *       Filename:  robust_pair_trader.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/10/2019 01:19:21 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "robust_pair_trader.h"
#include "pyrunner/pyrunner.h"
#include "basics/utils.h"
using namespace std;
using namespace athena;
Message
RobustPairTrader::processMsg(Message& msg) {
    Message outmsg;
    FXAct action = (FXAct)msg.getAction();
    switch(action) {
    case FXAct::ASK_PAIR:
        outmsg = procMsg_ASK_PAIR(msg);
        break;
    case FXAct::PAIR_HIST_X:
        outmsg = procMsg_PAIR_HIST_X(msg);
        break;
    case FXAct::PAIR_HIST_Y:
        outmsg = procMsg_PAIR_HIST_Y(msg);
        break;
    case FXAct::PAIR_MIN_OPEN:
        outmsg = procMsg_PAIR_MIN_OPEN(msg);
        break;
    case FXAct::NUM_POS:
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            real32* pm = (real32*)msg.getData();
            m_numPos = pm[0];
        });
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
    case FXAct::CLOSE_ALL_POS:
        Log(LOG_WARNING) << "Close all positions!";
        break;
    default:
        break;
    }

    return outmsg;
}

Message
RobustPairTrader::procMsg_ASK_PAIR(Message& msg) {
    String s1 = m_cfg->getSymX();
    String s2 = m_cfg->getSymY();
    String st = s1 + ":" + s2;
    Message outmsg(FXAct::ASK_PAIR,0,st.size());
    outmsg.setComment(st);

    Log(LOG_INFO) << "Sym pair: " + s1 + "," + s2;
    return outmsg;
}

Message
RobustPairTrader::procMsg_PAIR_HIST_X(Message& msg) {
    Log(LOG_INFO) << "X history arrives";

    SerializePack pack;
    unserialize(msg.getComment(),pack);

    int nbars = pack.int32_vec[0];
    int bar_size = pack.int32_vec[1];

    real32* pm = &pack.real32_vec[0];
    for (int i=0; i < nbars; i++) {
        m_openX.push_back(pm[0]);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History of X loaded: " + to_string(m_openX.size());

    Message out;
    return out;
}

Message
RobustPairTrader::procMsg_PAIR_HIST_Y(Message& msg) {
    Log(LOG_INFO) << "Y history arrives";

    SerializePack pack;
    unserialize(msg.getComment(),pack);

    int nbars = pack.int32_vec[0];
    int bar_size = pack.int32_vec[1];

    real32* pm = &pack.real32_vec[0];
    for (int i=0; i < nbars; i++) {
        m_openY.push_back(pm[0]);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History of Y loaded: " + to_string(m_openY.size());

    if (m_openX.size() != m_openY.size())
        Log(LOG_FATAL) << "Inconsistent length of X & Y";

    real64 corr = computePairCorr(m_openX,m_openY);
    Log(LOG_INFO) << "Correlation: " + to_string(corr);

    linreg();

    Message outmsg(msg.getAction(),sizeof(real32),0);
    pm = (real32*)outmsg.getData();
    pm[0] = m_currStatus["c1"];
    return outmsg;
}

void
RobustPairTrader::estimateSpreadTend(real64* sp, int len)
{
    real64 c0 = m_currStatus["c0"];
    real64 sigma = m_currStatus["sigma"];
    int n_up = 0;
    int n_down = 0;
    real64 h_dev = 0.;
    real64 l_dev = -0.;
    for (int i=0; i < len; i++) {
        real64 dev  = (sp[i] - c0)/sigma;
        if (dev >= 2) n_up++;
        if (dev <= -2) n_down++;
        if (dev > h_dev) h_dev = dev;
        if (dev < l_dev) l_dev = dev;
    }

    Log(LOG_INFO) << "Spread up ratio: " + to_string(n_up) + "/" + to_string(len);
    Log(LOG_INFO) << "Spread down ratio: " + to_string(n_down) + "/" + to_string(len);
    Log(LOG_INFO) << "Ratio range: [" + to_string(l_dev) + "," + to_string(h_dev) + "]";
}

void
RobustPairTrader::linreg(size_t start) {
    m_LRParams = robLinreg(m_openX,m_openY,start);

    real64 c0 = m_LRParams.c0;
    real64 c1 = m_LRParams.c1;
    Log(LOG_INFO) << "Linear reg done: c0 = " + to_string(c0)
                  + ", c1 = " + to_string(c1);

    Log(LOG_INFO) << "Ave weight: " + to_string(m_LRParams.w_ave);
    Log(LOG_INFO) << "Latest weight: " + to_string(m_LRParams.w_now);

    m_currStatus["c0"] = c0;
    m_currStatus["c1"] = c1;
    m_currStatus["rms"]= m_LRParams.rms;
    m_currStatus["r2"] = m_LRParams.r2;
    m_currStatus["sigma"] = m_LRParams.sigma;

    size_t len = m_openX.size() - start;
    real64* spread = new real64[len];
    int k=0;
    for (size_t i=start; i < m_openX.size(); i++) {
        spread[k++] = m_openY[i] - c1*m_openX[i];
    }

    real64 pv = testADF(spread,len);
    Log(LOG_INFO) << "p-value of non-stationarity of spread: " + to_string(pv);
    m_currStatus["pv"] = pv;

    real64 h = hurst(spread,len);
    Log(LOG_INFO) << "Hurst exponent: " + to_string(h);
    m_currStatus["hurst"] = h;

    real64 hf = compHalfLife(spread,len);
    Log(LOG_INFO) << "Half life of spread: " + to_string(hf);

    estimateSpreadTend(spread,len);

    vector<real64> vsp(spread,spread+len);
    dumpVectors("spread.csv",vsp);

    delete[] spread;
}

Message
RobustPairTrader::procMsg_PAIR_MIN_OPEN(Message& msg) {
    // This function fits linear model with all past ticks even though they are outliers
    Message outmsg(sizeof(real32),0);
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    real32* pm = &pack.real32_vec[0];
    real64 x = pm[0];
    real64 y = pm[1];
    real32 y_pv = pm[2];
    real32 y_pd = pm[3];
    m_openX.push_back(x);
    m_openY.push_back(y);

    dumpVectors("ticks.csv",m_openX,m_openY);

    char* pc = (char*)msg.getChar() + sizeof(int)*2;
    int cb = msg.getCharBytes() - sizeof(int)*2;
    String timestr = String(pc,cb);
    Log(LOG_INFO)<<"";

    Log(LOG_INFO) << "Mt5 time: " + timestr + ", X: " + to_string(x)
                  + ", Y: " + to_string(y);

    real64 corr = computePairCorr(m_openX,m_openY);
    Log(LOG_INFO) << "Correlation so far: " + to_string(corr);

    linreg(m_openX.size()-1000);

    Log(LOG_INFO) << "R2 = " + to_string(m_currStatus["r2"]);

    real32 sigma = m_currStatus["sigma"];
    Log(LOG_INFO) << "std = " + to_string(sigma);

    m_currStatus["rms"] = m_currStatus["rms"]/y_pv*y_pd;
    Log(LOG_INFO) << "rms = $" + to_string(m_currStatus["rms"]) + " (per unit volume)";

    real64 err = y - (m_currStatus["c0"] + m_currStatus["c1"]* x);
    real64 fac = err/m_currStatus["sigma"];
    Log(LOG_INFO) << " ====== err/std: " + to_string(fac) + " ======";

    pm = (real32*)outmsg.getData();
    pm[0] = m_currStatus["c1"];

    vector<real32> thd = m_cfg->getThresholdStd();

    if ( fac > thd[0] && fac < thd[1]) {
        outmsg.setAction(FXAct::PLACE_SELL);
    } else if( fac < -thd[0] && fac > -thd[1]) {
        outmsg.setAction(FXAct::PLACE_BUY);
    } else {
        outmsg.setAction(FXAct::NOACTION);
    }

    real64 w_ratio = m_LRParams.w_now/m_LRParams.w_ave;
    if (w_ratio < m_cfg->getOutlierWeightRatio()) {
        //outmsg.setAction(FXAct::NOACTION);
        m_numOutliers++;
        Log(LOG_WARNING) << "Outlier detected! w_now/w_ave: " + to_string(w_ratio);
    } else {
        m_numOutliers = 0;
    }

    if (m_numOutliers > 0) {
        Log(LOG_WARNING) << "So far outliers: " + to_string(m_numOutliers);
    }

    if (m_numOutliers >= m_cfg->getOutlierNumLimit()) {
        outmsg.setAction(FXAct::CLOSE_ALL_POS);
    }

    if(m_currStatus["pv"] > m_cfg->getStationaryPVLimit()) {
        outmsg.setAction(FXAct::NOACTION);
    }

    if(m_LRParams.r2 < m_cfg->getR2Limit()) {
        Log(LOG_WARNING) << "R2 too low";
        outmsg.setAction(FXAct::NOACTION);
    }

    return outmsg;
}
