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
#include "mean_revert/mean_revert.h"
#include "spread_trend/spread_trend.h"
#include <sstream>
using namespace std;
using namespace athena;

MinbarPairTrader::MinbarPairTrader(const String& cfg) : ServerBaseApp(cfg),m_isRunning(true),m_pairCount(0),m_oracle(nullptr) {
    m_cfg = &MptConfig::getInstance();
    m_cfg->loadConfig(cfg);
    m_initBalance = -1.;
    m_numPos = 0;

    m_oracle = new MeanRevert(this);
    Log(LOG_INFO) << "Minbar pair trader created with MeanRevert";
}

MinbarPairTrader::~MinbarPairTrader() {
    dumpVectors("prices.csv",m_x_ask,m_y_ask);
    dumpVectors("spreads.csv",m_spreads);
    if(m_oracle)
        delete m_oracle;

    Log(LOG_INFO) << "c0 = " + to_string(m_linParam.c0) + ", c1 = " + to_string(m_linParam.c1);
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
        m_x_ask.push_back(log(pm[0]));
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History of X loaded: " + to_string(m_x_ask.size());

    Message out;
    return out;
}

void
MinbarPairTrader::compOldSpreads() {
    size_t len = m_x_ask.size();
    real64* x = new real64[len];
    real64* y = new real64[len];

    for (size_t i=0; i< len; i++) {
        x[i] = m_x_ask[i];
        y[i] = m_y_ask[i];
    }

    //m_linParam = linreg(x,y,len);
    m_linParam = robLinreg(x,y,len);

    Log(LOG_INFO) << "Liner regression done. c0: " + to_string(m_linParam.c0) + ", c1: " + to_string(m_linParam.c1);

    for(int i=0; i < len; i++) {
        real64 tmp = y[i] - (m_linParam.c1 * x[i] + m_linParam.c0);
        m_spreads.push_back(tmp);
    }

    real64 r2 = compR2(m_linParam,x,y,len);

    Log(LOG_INFO) << "R2: " + to_string(r2);
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
        m_y_ask.push_back(log(pm[0]));
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History of Y loaded: " + to_string(m_y_ask.size());

    if ( m_x_ask.size() != m_y_ask.size() )
        Log(LOG_FATAL) << "Inconsistent length of X & Y";

    real64 corr = computePairCorr(m_x_ask, m_y_ask);
    Log(LOG_INFO) << "Correlation: " + to_string(corr);

    compOldSpreads();

    real64 pv = testADF(&m_spreads[0],m_spreads.size());
    Log(LOG_INFO) << "p-value of stationarity of spreads: " + to_string(pv);

    m_oracle->init();

    Message outmsg(msg.getAction(), sizeof(real32), 0);
    pm = (real32*)outmsg.getData();
    pm[0] = m_linParam.c1;

    if (m_linParam.c1 > 0) {
        m_posPairDirection = OPPOSITE;
    } else {
        m_posPairDirection = SAME;
    }
    return outmsg;
}

real64
MinbarPairTrader::compSpread(real64 x, real64 y) {
    real64 err = y - (m_linParam.c0 + m_linParam.c1*x);
    return err;
}

Message
MinbarPairTrader::procMsg_PAIR_MIN_OPEN(Message& msg) {
    m_pairCount++;
    Message outmsg(sizeof(real32), 0);
    real32* pm = (real32*)msg.getData();
    real64 x_ask = log(pm[0]);
    real64 x_bid = log(pm[1]);
    real64 y_ask = log(pm[2]);
    real64 y_bid = log(pm[3]);
    m_x_ask.push_back(x_ask);
    m_x_bid.push_back(x_bid);
    m_y_ask.push_back(y_ask);
    m_y_bid.push_back(y_bid);

    ostringstream oss;
    oss << "\n\t" << m_pairCount << "th pair arrives. x_ask: " << x_ask << ", x_bid: " << x_bid << ", y_ask: " << y_ask << ", y_bid: " << y_bid;
    Log(LOG_INFO) << oss.str();

    switch(m_posPairDirection) {
    case SAME: {
        SpreadInfo buy_y_spread;
        buy_y_spread.create = compSpread(x_ask,y_ask);
        buy_y_spread.close  = compSpread(x_bid,y_bid);
        SpreadInfo sell_y_spread;
        sell_y_spread.create = compSpread(x_bid,y_bid);
        sell_y_spread.close  = compSpread(x_ask,y_ask);
        m_buy_y_spreads.push_back(buy_y_spread);
        m_sell_y_spreads.push_back(sell_y_spread);
    }
        break;
    case OPPOSITE: {
        SpreadInfo buy_y_spread;
        buy_y_spread.create = compSpread(x_bid,y_ask);
        buy_y_spread.close  = compSpread(x_ask,y_bid);
        SpreadInfo sell_y_spread;
        sell_y_spread.create = compSpread(x_ask,y_bid);
        sell_y_spread.close  = compSpread(x_bid,y_ask);
        m_buy_y_spreads.push_back(buy_y_spread);
        m_sell_y_spreads.push_back(sell_y_spread);
    }
        break;
    default:
        Log(LOG_FATAL) << "unknown position pair directions";
        break;
    }

    FXAct act = m_oracle->getDecision();
    outmsg.setAction(act);
    pm = (real32*)outmsg.getData();
    pm[0] = m_linParam.c1;

    m_isRunning = m_oracle->isContinue();
    if(!m_isRunning)
        outmsg.setAction(FXAct::NOACTION);

#if 0
    dumpVectors("ticks.csv",m_x_ask, m_y_ask);

    char* pc = (char*)msg.getChar() + sizeof(int)*2;
    int cb = msg.getCharBytes() - sizeof(int)*2;
    String timestr = String(pc, cb);
    Log(LOG_INFO) << "";

    Log(LOG_INFO) << "Mt5 time: " + timestr + ", X: " + to_string(x)
                  + ", Y: " + to_string(y);

    real64 corr = computePairCorr(m_x_ask, m_y_ask);
    Log(LOG_INFO) << "Correlation so far: " + to_string(corr);

    linreg(m_x_ask.size()-1000);

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
