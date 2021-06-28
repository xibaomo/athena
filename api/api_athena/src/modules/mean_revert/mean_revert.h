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

class MeanRevert : public DecisionMaker {
private:
    real64 m_devUnit; // unit of spread deviation from preset mean
    std::vector<SpreadInfo> m_spreadDevs;

    real64 m_highBuyDev, m_lowBuyDev; // in unit of devUnit
    real64 m_highSellDev, m_lowSellDev; // in unit of devUnit

    size_t m_buys, m_sells, m_numclose;

    real64 m_curMean;
    bool   m_isUpdateMean;

    std::vector<real64> m_cuScores;
public:
    MeanRevert(MinbarPairTrader* p) : DecisionMaker(p), m_buys(0), m_sells(0), m_numclose(0), m_isUpdateMean(false){;}
    ~MeanRevert();
    void init();

    // find median of deviation from spread mean. use it as std;
    // regular std may be extended too much if extreme cases happen.
    real64 findMedianDev(const std::vector<real64>& spreads, const real64 mean);

    real64 compLatestSpreadMA();

    // mean value insofar
    real64 compLatestSpreadMean(size_t);

    // -1 means not time to do stats
    int stats();

    void dumpDevs(const String& fn);
    FXAct getDecision();

    bool isContinue() { return true; }
};
