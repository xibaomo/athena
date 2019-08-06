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

typedef std::unordered_map<String,real32> PairStatus;

class MinBarPairTrader : public ServerBaseApp {
protected:

    std::unordered_map<String,std::vector<real32>> m_sym2hist;

    std::vector<MinBar> m_minbarX;
    std::vector<MinBar> m_minbarY;

    std::vector<real32> m_openX;
    std::vector<real32> m_openY;

    std::unordered_map<String,PairStatus> m_pairTracker;

    PairStatus m_currStatus;

    LRParam             m_linregParam;

    real64 m_spreadMean;
    real64 m_spreadStd;

    real64 m_initSpreadMean;
    real64 m_initSpreadStd;

    real64 m_maxStatPValue;

    MptConfig* m_cfg;
    real64 m_prevSpread;
    MinBarPairTrader(const String& cf) : ServerBaseApp (cf){
        m_cfg = &MptConfig::getInstance();
        m_cfg->loadConfig(cf);
        m_prevSpread = 0.;
        m_spreadMean = -1.;
        m_spreadStd = -1.;
        m_initSpreadMean = m_spreadMean;
        m_initSpreadStd  = m_spreadStd;
        m_maxStatPValue  = -1.;

    }

public:
    virtual ~MinBarPairTrader() {;}
    static MinBarPairTrader& getInstance(const String& cf) {
        static MinBarPairTrader _ins(cf);
        return _ins;
    }

    void prepare() {;}
    void finish();

    void dumpStatus();
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

    Message procMsg_PAIR_POS_PLACED(Message& msg);
    Message procMsg_PAIR_POS_CLOSED(Message& msg);

    //real64 computePairCorr(std::vector<real32>& v1, std::vector<real32>& v2);

    /**
     * Linear regression of X & Y
     */
    void linearReg(int start);

    template <typename T>
    std::vector<T> generateBins(T min, T max, T mean, T sd, T bin_size_sd);

    void compDistr(real64* data, int len, real64* mean=nullptr, real64* sd=nullptr);
    void selectTopCorr();

};

template <typename T>
std::vector<T>
MinBarPairTrader::generateBins(T min, T max, T mean, T sd, T bin_size_sd)
{
    const T eps=1e-6;
    std::vector<T> neg_bin;
    T s = mean;
    while (1) {
        s -= bin_size_sd * sd;
        if (s < min) {
            neg_bin.push_back(min-eps);
            break;
        }
        neg_bin.push_back(s);
    }
    std::reverse(neg_bin.begin(),neg_bin.end());

    std::vector<T> pos_bin;
    s = mean;
    while(1) {
        pos_bin.push_back(s);
        s+=bin_size_sd*sd;
        if (s > max) {
            pos_bin.push_back(max+eps);
            break;
        }
    }

    neg_bin.insert(neg_bin.end(),pos_bin.begin(),pos_bin.end());

    return neg_bin;
}
#endif   /* ----- #ifndef _SERVER_APP_MINBAR_PAIR_TRADER_H_  ----- */
