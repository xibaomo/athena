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
#include "linreg/roblinreg.h"

typedef std::unordered_map<String, real32> PTStatus;

class MinbarPairTrader : public ServerBaseApp {
protected:
    real32                  m_initBalance;
    MptConfig*              m_cfg;
    std::vector<real32>     m_openX;
    std::vector<real32>     m_openY;
    PTStatus                m_currStatus;
    size_t                  m_numPos;

    size_t                  m_numOutliers;

    RobLRParam              m_LRParams;

    MinbarPairTrader(const String& cfg) : ServerBaseApp(cfg) {
        m_cfg = &MptConfig::getInstance();
        m_cfg->loadConfig(cfg);
        m_initBalance = -1.;
        m_numPos = 0;
        m_numOutliers = 0;
        Log(LOG_INFO) << "Robust pair trader created";
        }
public:
    virtual ~MinbarPairTrader() {;}

    static MinbarPairTrader& getInstance(const String& cfg) {
        static MinbarPairTrader _ins(cfg);
        return _ins;
    }

    void prepare() {;}

    Message processMsg(Message& msg);
    Message procMsg_ASK_PAIR(Message& msg);
    Message procMsg_PAIR_HIST_X(Message& msg);
    Message procMsg_PAIR_HIST_Y(Message& msg);
    Message procMsg_PAIR_MIN_OPEN(Message& msg);

    void linreg(size_t start = 0);

    void estimateSpreadTend(real64* sp, int len);
};

