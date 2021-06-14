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
enum PosPairDir {
    NONE,
    SAME,
    OPPOSITE
};

struct SpreadInfo {
    real64 buy;
    real64 sell;
};
class DecisionMaker;
class MinbarPairTrader : public ServerBaseApp {
  protected:
    real32                  m_initBalance;
    MptConfig*              m_cfg;
    std::vector<real64>     m_x_ask;
    std::vector<real64>     m_y_ask;
    std::vector<real64>     m_x_bid;
    std::vector<real64>     m_y_bid;
    std::vector<real64>     m_spreads; // mid of buy and sell spreads
    std::vector<SpreadInfo>     m_tradeSpreads; // from trading period
    size_t                  m_numPos;
    bool                    m_isRunning;
    size_t                  m_pairCount;
    PosPairDir              m_posPairDirection;

    RobLRParam       m_linParam;

    DecisionMaker* m_oracle;

    MinbarPairTrader(const String& cfg);
  public:
    virtual ~MinbarPairTrader();

    static MinbarPairTrader& getInstance(const String& cfg) {
        static MinbarPairTrader _ins(cfg);
        return _ins;
    }

    MptConfig* getConfig() {
        return m_cfg;
    }
    void prepare() {;}

    void compOldSpreads();
    std::vector<real64>& getSpreads() {
        return m_spreads;
    }
    std::vector<real64>& getOpenX() {
        return m_x_ask;
    }
    std::vector<real64>& getOpenY() {
        return m_y_ask;
    }

    SpreadInfo getLatestSpread() {
        return m_tradeSpreads.back();
    }

    void dumpTradeSpreads();

    std::vector<SpreadInfo>& getTradeSpreads() { return m_tradeSpreads; }

    PosPairDir getPosPairDir() { return m_posPairDirection; }
    real64 compSpread(real64 x, real64 y);

    Message processMsg(Message& msg);
    Message procMsg_ASK_PAIR(Message& msg);
    Message procMsg_PAIR_HIST_X(Message& msg);
    Message procMsg_PAIR_HIST_Y(Message& msg);
    Message procMsg_PAIR_MIN_OPEN(Message& msg);
};

