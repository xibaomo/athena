/*
 * =====================================================================================
 *
 *       Filename:  multi_pairtrader.h
 *
 *    Description:  Pair trader of multiple pairs
 *
 *        Version:  1.0
 *        Created:  06/16/2019 06:53:28 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MULTI_PAir_TRADER_H_
#define  _MULTI_PAir_TRADER_H_
#include "server_apps/server_base_app/server_base_app.h"
#include "linreg/linreg.h"
#include "mulpairconf.h"
#include <unordered_map>
struct SymPair {
    String symx;
    String symy;
    real64 corr;
};
class MultiPairTrader : public ServerBaseApp {
protected:
    MultiPairConfig* m_cfg;

    std::unordered_map<String,std::vector<real32>> m_sym2hist;
    std::vector<SymPair> m_topCorrSyms;

    MultiPairTrader(const String& cf) : ServerBaseApp(cf) {
        m_cfg = &MultiPairConfig::getInstance();
        m_cfg->loadConfig(cf);
    }
public:
    virtual ~MultiPairTrader() {;}

    static MultiPairTrader& getInstance(const String& cf) {
        static MultiPairTrader _ins(cf);
        return _ins;
    }

    void prepare() {;}

    Message processMsg(Message& msg);
    Message procMsg_SYM_HIST_OPEN(Message& msg);
    Message procMsg_ASK_PAIR(Message& msg);

    bool test_coint(std::vector<real32>& v1, std::vector<real32>& v2);

    void selectTopCorr();

};
#endif   /* ----- #ifndef _MULTI_PAir_TRADER_H_  ----- */
