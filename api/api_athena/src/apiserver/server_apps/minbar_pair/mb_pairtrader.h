/*
 * =====================================================================================
 *
 *       Filename:  mb_pairtrader.h
 *
 *    Description:
 *
 *
 *        Version:  1.0
 *        Created:  05/26/2019 04:39:06 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _SERVER_APP_MINBAR_PAIR_TRADER_H_
#define  _SERVER_APP_MINBAR_PAIR_TRADER_H_

#include "server_apps/server_base_app/server_base_app.h"
#include "mptconf.h"
#include "linreg/linreg.h"
#include <unordered_map>

class MinBarPairTrader : public ServerBaseApp {
protected:

    std::unordered_map<String,std::vector<real32>> m_sym2hist;

    std::vector<MinBar> m_minbarX;
    std::vector<MinBar> m_minbarY;

    std::vector<real32> m_openX;
    std::vector<real32> m_openY;

    LRParam             m_linregParam;

    real64 m_spreadMean;
    real64 m_spreadStd;

    MptConfig* m_cfg;
    real64 m_prevSpread;
    MinBarPairTrader(const String& cf) : ServerBaseApp (cf){
        m_cfg = &MptConfig::getInstance();
        m_cfg->loadConfig(cf);
        m_prevSpread = 0.;
    }

public:
    virtual ~MinBarPairTrader() {;}
    static MinBarPairTrader& getInstance(const String& cf) {
        static MinBarPairTrader _ins(cf);
        return _ins;
    }

    void prepare() {;}
    Message processMsg(Message& msg);

    Message procMsg_ASK_PAIR(Message& msg);
    void loadHistoryFromMsg(Message& msg, std::vector<MinBar>& v, std::vector<real32>& openvec);

    Message procMsg_PAIR_MIN_OPEN(Message& msg);

    /**
     * Load min bar history for Y
     * Send back hedge factor
     */
    Message procMsg_PAIR_HIST_Y(Message& msg);

    Message procMsg_SYM_HIST_OPEN(Message& msg);

    //real64 computePairCorr(std::vector<real32>& v1, std::vector<real32>& v2);

    /**
     * Linear regression of X & Y
     */
    void linearReg();

    void selectTopCorr();

    bool test_coint(std::vector<real32>& v1, std::vector<real32>& v2);
};
#endif   /* ----- #ifndef _SERVER_APP_MINBAR_PAIR_TRADER_H_  ----- */
