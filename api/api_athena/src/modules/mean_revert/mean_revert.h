/*
 * =====================================================================================
 *
 *       Filename:  mean_revert.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/11/2021 11:24:51 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once

#include "minbar_pair_trader/decision_maker.h"
#include "minbar_pair_trader/minbar_pair_trader.h"
#include "linreg/linreg.h"
class MeanRevert : public DecisionMaker {
    enum PosPairDir {
        NONE,
        SAME,
        OPPOSITE
    };
    struct SpreadInfo {
        real64 buy;
        real64 sell;
    };
  private:
    real64                  m_devUnit; // unit of spread deviation from preset mean
    std::vector<SpreadInfo> m_spreadDevs;

    real64                  m_highBuyDev, m_lowBuyDev; // in unit of devUnit
    real64                  m_highSellDev, m_lowSellDev; // in unit of devUnit

    size_t                  m_buys, m_sells, m_numclose;

    PosPairDir              m_posPairDirection;

    LRParam                 m_linParam;
    std::vector<real64>     m_spreads;
    std::vector<SpreadInfo> m_tradeSpreads;

    real64                  m_curMean;

    std::vector<real64>     m_cuScores;
  public:
    MeanRevert(MinbarPairTrader* p) : DecisionMaker(p), m_buys(0), m_sells(0), m_numclose(0){;}
    ~MeanRevert();
    void init();

    real64 getSlope() { return m_linParam.c1; }

    // find median of deviation from spread mean. use it as std;
    // regular std may be extended too much if extreme cases happen.
    real64 findMedianDev(const std::vector<real64>& spreads, const real64 mean);

    void compOldSpreads(); // compute spreads of old history
    void compNewSpreads(); // compute the latest buy, sell, mid spreads

    real64 compLatestSpreadMA();

    void updateModel(int hist_len);

    // mean value insofar
    real64 compLatestSpreadMean(size_t);

    void compDevFromMean();

    real64 compSpread(real64 x, real64 y);

    void dumpDevs(const String& fn);
    FXAct getDecision();

    bool isContinue() { return true; }

    void dumpTradeSpreads();

    std::vector<SpreadInfo>& getTradeSpreads() { return m_tradeSpreads; }
};
