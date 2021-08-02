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
#include "pair_asset_meanrevert/pair_asset_meanrevert.h"
#include <sstream>
using namespace std;
using namespace athena;

MinbarPairTrader::MinbarPairTrader(const String& cfg) : ServerBaseApp(cfg), m_curNumPos(0), m_isRunning(true), m_pairCount(0), m_maxProfit(-1.f),
                                                        m_avgProfit(-1.f),m_oracle(nullptr) {
    m_cfg = &MptConfig::getInstance();
    m_cfg->loadConfig(cfg);
    m_initBalance = -1.;

    m_oracle = new PairAssetMeanRevert(this);
    Log(LOG_INFO) << "Minbar pair trader created.";
}

MinbarPairTrader::~MinbarPairTrader() {
    dumpVectors("prices.csv",m_mid_x, m_mid_y);
    dumpVectors("assets.csv",m_assetX_mid,m_assetY_mid);
    dumpVectors("trade_assets.csv",m_assetX_buy,m_assetX_sell,m_assetY_buy,m_assetY_sell);

    if ( m_oracle )
        delete m_oracle;

    Log(LOG_INFO) << "Max profit: " + to_string(m_maxProfit) + ", max avg/pos: " + to_string(m_avgProfit);
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
    case FXAct::ACCOUNT_BALANCE:
        outmsg = procMsg_noreply(msg,[&](Message& msg) {
            real64* pm = (real64*)msg.getData();
            Log(LOG_INFO) << "Account balance: " + to_string(pm[0]);
        });
        break;
    case FXAct::GET_LOTS:
        outmsg = procMsg_GOT_LOTS(msg);
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
    case FXAct::CLOSE_BUY:
        Log(LOG_INFO) << "Action: close buy positions";
        break;
    case FXAct::CLOSE_SELL:
        Log(LOG_INFO) << "Action: close sell positions";
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

    SerializePack pack;
    unserialize(msg.getComment(), pack);

    int nbars = pack.int32_vec[0];
    int bar_size = pack.int32_vec[1];

    auto& v = pack.real32_vec;
    m_ticksize_x = pack.real64_vec[0];
    m_tickval_x  = pack.real64_vec[1];
    size_t pm = 0;
    for ( int i = 0; i < nbars; i++ ) {
        m_mid_x.push_back(v[pm]);
        m_assetX_mid.push_back(v[pm]/m_ticksize_x*m_tickval_x);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History of X loaded: " + to_string(m_mid_x.size());

    Message out;
    return out;
}

Message
MinbarPairTrader::procMsg_PAIR_HIST_Y(Message& msg) {
    Log(LOG_INFO) << "Y history arrives";

    SerializePack pack;
    unserialize(msg.getComment(), pack);

    int nbars = pack.int32_vec[0];
    int bar_size = pack.int32_vec[1];

    m_ticksize_y = pack.real64_vec[0];
    m_tickval_y  = pack.real64_vec[1];
    auto& v = pack.real32_vec;
    size_t pm = 0;
    for ( int i = 0; i < nbars; i++ ) {
        m_mid_y.push_back(v[pm]);
        m_assetY_mid.push_back(v[pm]/m_ticksize_y*m_tickval_y);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History of Y loaded: " + to_string(m_mid_y.size());

    if ( m_mid_x.size() != m_mid_y.size() )
        Log(LOG_FATAL) << "Inconsistent length of X & Y";

    real64 corr = computePairCorr(m_mid_x, m_mid_y);
    Log(LOG_INFO) << "Correlation: " + to_string(corr);


    m_oracle->init();
    m_oracle->setLookback(m_mid_x.size());


    Message outmsg(msg.getAction(), sizeof(real32), 0);
    real32* p = (real32*)outmsg.getData();
    p[0] = -999.f;

    return outmsg;
}

Message
MinbarPairTrader::procMsg_PAIR_MIN_OPEN(Message& msg) {
    m_pairCount++;
    Message outmsg(sizeof(real32), 0);
    SerializePack pack;
    unserialize(msg.getComment(), pack);

    real64 x_ask = pack.real64_vec[0];
    real64 x_bid = pack.real64_vec[1];
    m_ticksize_x = pack.real64_vec[2];
    m_tickval_x  = pack.real64_vec[3];
    real64 y_ask = pack.real64_vec1[0];
    real64 y_bid = pack.real64_vec1[1];
    m_ticksize_y = pack.real64_vec1[2];
    m_tickval_y  = pack.real64_vec1[3];
    m_x_ask.push_back(x_ask);
    m_x_bid.push_back(x_bid);
    m_y_ask.push_back(y_ask);
    m_y_bid.push_back(y_bid);
    real64 midx, midy;
    midx = (x_ask+x_bid)*.5f;
    midy = (y_ask+y_bid)*.5f;
    m_mid_x.push_back((midx));
    m_mid_y.push_back((midy));
    m_assetX_mid.push_back(midx/m_ticksize_x*m_tickval_x);
    m_assetY_mid.push_back(midy/m_ticksize_y*m_tickval_y);

    m_assetX_buy.push_back(x_ask/m_ticksize_x*m_tickval_x);
    m_assetX_sell.push_back(x_bid/m_ticksize_x*m_tickval_x);
    m_assetY_buy.push_back(y_ask/m_ticksize_y*m_tickval_y);
    m_assetY_sell.push_back(y_bid/m_ticksize_y*m_tickval_y);

    m_curNumPos = pack.int32_vec[0];

    Log(LOG_INFO) << "Mt5 time: " + pack.str_vec[0];
    ostringstream oss;
    oss << "\n\t" << m_pairCount << "th pair arrives. x_ask: " << x_ask << ", x_bid: " << x_bid << ", y_ask: " << y_ask << ", y_bid: " << y_bid;
    Log(LOG_INFO) << oss.str();
    Log(LOG_INFO) << "tick val: x: " + to_string(m_tickval_x) + ", y: " + to_string(m_tickval_y);
    Log(LOG_INFO) << "Num of positions: " + to_string(m_curNumPos);
    Log(LOG_INFO) << "Num of take-profit: " + to_string(pack.int32_vec[1]) + ", stop-loss: " + to_string(pack.int32_vec[2]);

    real64 profit= pack.real64_vec1[4];
    real64 avg_profit = profit / m_curNumPos;
    Log(LOG_INFO) << "Current profit: " + to_string(profit) + ", avg/pos: " + to_string(avg_profit);
    if (m_maxProfit < profit) { m_maxProfit = profit;}
    if (m_avgProfit < avg_profit) { m_avgProfit = avg_profit; }

    FXAct act = m_oracle->getDecision();

    outmsg.setAction(act);
    real32* p = (real32*)outmsg.getData();
    p[0] = m_oracle->getSlope();

    m_isRunning = m_oracle->isContinue();
    if ( !m_isRunning )
        outmsg.setAction(FXAct::NOACTION);

    return outmsg;
}

Message
MinbarPairTrader::procMsg_GOT_LOTS(Message& msg) {
    real64 lx,ly;
    Message outmsg(sizeof(real64)*2);
    real64* pm = (real64*)outmsg.getData();

    m_oracle->getBestLots(lx,ly);
    pm[0] = lx;
    pm[1] = ly;
    outmsg.setAction(FXAct::GET_LOTS);

    return outmsg;
}
