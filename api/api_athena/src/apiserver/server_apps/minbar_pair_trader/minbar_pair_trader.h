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

typedef std::unordered_map<String, real32> PTStatus;

class DecisionMaker;
class MinbarPairTrader : public ServerBaseApp {
protected:
    real32                  m_initBalance;
    MptConfig*              m_cfg;
    std::vector<real32>     m_openX;
    std::vector<real32>     m_openY;
    std::vector<real64>     m_errs;
    PTStatus                m_currStatus;
    size_t                  m_numPos;

    LRParam       m_linParam;

    DecisionMaker* m_oracle;


    MinbarPairTrader(const String& cfg) : ServerBaseApp(cfg),m_oracle(nullptr) {
        m_cfg = &MptConfig::getInstance();
        m_cfg->loadConfig(cfg);
        m_initBalance = -1.;
        m_numPos = 0;
        Log(LOG_INFO) << "Minbar pair trader created";
        }
public:
    virtual ~MinbarPairTrader() {;}

    static MinbarPairTrader& getInstance(const String& cfg) {
        static MinbarPairTrader _ins(cfg);
        return _ins;
    }

    void prepare() {;}

    void compErrs();
    Message processMsg(Message& msg);
    Message procMsg_ASK_PAIR(Message& msg);
    Message procMsg_PAIR_HIST_X(Message& msg);
    Message procMsg_PAIR_HIST_Y(Message& msg);
    Message procMsg_PAIR_MIN_OPEN(Message& msg);
};

