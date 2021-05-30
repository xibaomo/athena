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

#include "minbar_pair_trader.h"
#include "pyrunner/pyrunner.h"
#include "basics/utils.h"
#include "linreg/linreg.h"
#include "dm_rule_75.h"
using namespace std;
using namespace athena;

MinbarPairTrader::MinbarPairTrader(const String& cfg) : ServerBaseApp(cfg),m_oracle(nullptr) {
    m_cfg = &MptConfig::getInstance();
    m_cfg->loadConfig(cfg);
    m_initBalance = -1.;
    m_numPos = 0;

    m_oracle = new Rule75(this);
    Log(LOG_INFO) << "Minbar pair trader created";
}

Message
MinbarPairTrader::processMsg(Message& msg) {
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
        outmsg = procMsg_noreply(msg, [this](Message& msg) {
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
MinbarPairTrader::procMsg_ASK_PAIR(Message& msg) {
    String s1 = m_cfg->getSymX();
    String s2 = m_cfg->getSymY();
    String st = s1 + ":" + s2;
    Message outmsg(FXAct::ASK_PAIR, 0, st.size());
    outmsg.setComment(st);

    Log(LOG_INFO) << "Sym pair: " + s1 + "," + s2;
    return outmsg;
}

Message
MinbarPairTrader::procMsg_PAIR_HIST_X(Message& msg) {
    Log(LOG_INFO) << "X history arrives";

    int* pc = (int*)msg.getChar();
    int nbars = pc[0];
    int bar_size = pc[1];

    real32* pm = (real32*)msg.getData();
    for ( int i = 0; i < nbars; i++ ) {
        m_openX.push_back(pm[0]);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History of X loaded: " + to_string(m_openX.size());

    Message out;
    return out;
}

void
MinbarPairTrader::compErrs() {
    size_t len = m_openX.size();
    real64* x = new real64[len];
    real64* y = new real64[len];

    for (size_t i=0; i< len; i++) {
        x[i] = m_openX[i];
        y[i] = m_openY[i];
    }

    m_linParam = linreg(x,y,len);

    for(int i=0; i < len; i++) {
        real64 tmp = m_linParam.c0 + m_linParam.c1 * x[i] - y[i];
        m_errs.push_back(tmp);
    }

    delete[] x;
    delete[] y;
}

Message
MinbarPairTrader::procMsg_PAIR_HIST_Y(Message& msg) {
    Log(LOG_INFO) << "Y history arrives";

    int* pc = (int*)msg.getChar();
    int nbars = pc[0];
    int bar_size = pc[1];

    real32* pm = (real32*)msg.getData();
    for ( int i = 0; i < nbars; i++ ) {
        m_openY.push_back(pm[0]);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History of Y loaded: " + to_string(m_openY.size());

    if ( m_openX.size() != m_openY.size() )
        Log(LOG_FATAL) << "Inconsistent length of X & Y";

    real64 corr = computePairCorr(m_openX, m_openY);
    Log(LOG_INFO) << "Correlation: " + to_string(corr);

    compErrs();

    real64 pv = testADF(&m_errs[0],m_errs.size());

    Log(LOG_INFO) << "p-value of adf test on errors: " + to_string(pv);

    Message outmsg(msg.getAction(), sizeof(real32), 0);
    pm = (real32*)outmsg.getData();
    pm[0] = m_linParam.c1;
    return outmsg;
}

Message
MinbarPairTrader::procMsg_PAIR_MIN_OPEN(Message& msg) {
    Message outmsg(sizeof(real32), 0);
    real32* pm = (real32*)msg.getData();
    real64 x = pm[0];
    real64 y = pm[1];
    real32 y_pv = pm[2];
    real32 y_pd = pm[3];
    m_openX.push_back(x);
    m_openY.push_back(y);

    real64 err = m_linParam.c0 + m_linParam.c1*x - y;

    m_errs.push_back(err);

#if 0
    dumpVectors("ticks.csv",m_openX, m_openY);

    char* pc = (char*)msg.getChar() + sizeof(int)*2;
    int cb = msg.getCharBytes() - sizeof(int)*2;
    String timestr = String(pc, cb);
    Log(LOG_INFO) << "";

    Log(LOG_INFO) << "Mt5 time: " + timestr + ", X: " + to_string(x)
                  + ", Y: " + to_string(y);

    real64 corr = computePairCorr(m_openX, m_openY);
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

    //vector<real32> thd = m_cfg->getThresholdStd();

    if ( fac > thd[0] && fac < thd[1] ) {
        outmsg.setAction(FXAct::PLACE_SELL);
    } else if( fac < -thd[0] && fac > -thd[1]) {
        outmsg.setAction(FXAct::PLACE_BUY);
    } else {
        outmsg.setAction(FXAct::NOACTION);
    }

    real64 w_ratio = m_LRParams.w_now/m_LRParams.w_ave;
    if ( w_ratio < m_cfg->getOutlierWeightRatio() ) {
        //outmsg.setAction(FXAct::NOACTION);
        m_numOutliers++;
        Log(LOG_WARNING) << "Outlier detected! w_now/w_ave: " + to_string(w_ratio);
    } else {
        m_numOutliers = 0;
    }

    if ( m_numOutliers > 0 ) {
        Log(LOG_WARNING) << "So far outliers: " + to_string(m_numOutliers);
    }

    if ( m_numOutliers >= m_cfg->getOutlierNumLimit() ) {
        outmsg.setAction(FXAct::CLOSE_ALL_POS);
    }

    if ( m_currStatus["pv"] > m_cfg->getStationaryPVLimit() ) {
        outmsg.setAction(FXAct::NOACTION);
    }

    if ( m_LRParams.r2 < m_cfg->getR2Limit() ) {
        Log(LOG_WARNING) << "R2 too low";
        outmsg.setAction(FXAct::NOACTION);
    }
#endif // 0


    return outmsg;
}
