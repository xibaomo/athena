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

class MinBarPairTrader : public ServerBaseApp {
protected:

    std::vector<MinBar> m_minbarX;
    std::vector<MinBar> m_minbarY;

    LRParam             m_linregParam;

    MptConfig* m_cfg;
    MinBarPairTrader(const String& cf) : ServerBaseApp (cf){
        m_cfg = &MptConfig::getInstance();
        m_cfg->loadConfig(cf);
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
    void loadHistoryFromMsg(Message& msg, std::vector<MinBar>& v);

    Message procMsg_PAIR_MIN_OPEN(Message& msg);

    real64 computePairCorr();

    /**
     * Linear regression of X & Y
     */
    void linearReg();
};
#endif   /* ----- #ifndef _SERVER_APP_MINBAR_PAIR_TRADER_H_  ----- */
