/*
 * =====================================================================================
 *
 *       Filename:  minbar_pair_trader.h
 *
 *    Description:
 *        Version:  1.0
 *        Created:  08/10/2019 01:17:40 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once

#include "server_apps/server_base_app/server_base_app.h"
#include "mptconf.h"
#include "linreg/linreg.h"
#include "linreg/roblinreg.h"

class DecisionMaker;
class MinbarPairTrader : public ServerBaseApp {
  protected:
    real32              m_initBalance;
    MptConfig*          m_cfg;
    std::vector<real64> m_x_ask;
    std::vector<real64> m_y_ask;
    std::vector<real64> m_x_bid;
    std::vector<real64> m_y_bid;
    std::vector<real64> m_mid_x;
    std::vector<real64> m_mid_y;
    std::vector<real64> m_assetX;
    std::vector<real64> m_assetY;

    real64              m_ticksize_x;
    real64              m_tickval_x;
    real64              m_ticksize_y;
    real64              m_tickval_y;
    int                 m_curNumPos;
    bool                m_isRunning;
    size_t              m_pairCount;

    real64              m_maxProfit;
    real64              m_avgProfit;
    DecisionMaker*      m_oracle;

    MinbarPairTrader(const String& cfg);
  public:
    virtual ~MinbarPairTrader();

    static MinbarPairTrader& getInstance(const String& cfg) {
        static MinbarPairTrader _ins(cfg);
        return _ins;
    }

    MptConfig* getConfig()         { return m_cfg; }
    void prepare()                 {;}

    std::vector<real64>& getMidX() { return m_mid_x; }
    std::vector<real64>& getMidY() { return m_mid_y; }

    std::vector<real64>& getAskX()  { return m_x_ask; }
    std::vector<real64>& getAskY()  { return m_y_ask; }
    std::vector<real64>& getBidX()  { return m_x_bid; }
    std::vector<real64>& getBidY()  { return m_y_bid; }
    std::vector<real64>& getAssetX() { return m_assetX; }
    std::vector<real64>& getAssetY() { return m_assetY; }

    real64 getTickSizeX()          { return m_ticksize_x; }
    real64 getTickSizeY()          { return m_ticksize_y; }
    real64 getTickValX()           { return m_tickval_x; }
    real64 getTickValY()           { return m_tickval_y; }


    size_t getPairCount()          { return m_pairCount; }

    int getCurNumPos() const       { return m_curNumPos; }

    virtual Message processMsg(Message& msg);
    Message procMsg_ASK_PAIR(Message& msg);
    Message procMsg_PAIR_HIST_X(Message& msg);
    Message procMsg_PAIR_HIST_Y(Message& msg);
    virtual Message procMsg_PAIR_MIN_OPEN(Message& msg);
};

