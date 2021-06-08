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
class DecisionMaker;
class MinbarPairTrader : public ServerBaseApp {
  protected:
    real32                  m_initBalance;
    MptConfig*              m_cfg;
    std::vector<real64>     m_openX;
    std::vector<real64>     m_openY;
    std::vector<real64>     m_spreads;
    size_t                  m_numPos;
    bool                    m_isRunning;
    size_t                  m_pairCount;

    LRParam       m_linParam;

    DecisionMaker* m_oracle;

    MinbarPairTrader(const String& cfg);
  public:
    virtual ~MinbarPairTrader();

    static MinbarPairTrader& getInstance(const String& cfg) {
        static MinbarPairTrader _ins(cfg);
        return _ins;
    }

    MptConfig* getConfig() { return m_cfg; }
    void prepare() {;}

    void compSpreads();
    std::vector<real64>& getSpreads() { return m_spreads; }
    std::vector<real64>& getOpenX() {return m_openX;}
    std::vector<real64>& getOpenY() { return m_openY; }

    Message processMsg(Message& msg);
    Message procMsg_ASK_PAIR(Message& msg);
    Message procMsg_PAIR_HIST_X(Message& msg);
    Message procMsg_PAIR_HIST_Y(Message& msg);
    Message procMsg_PAIR_MIN_OPEN(Message& msg);
};

