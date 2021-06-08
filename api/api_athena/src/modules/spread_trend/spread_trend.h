/*
 * =====================================================================================
 *
 *       Filename:  spread_trend.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/04/2021 05:07:24 PM
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
#include "spread_trend/spread_trend_conf.h"
class SpreadTrend : public DecisionMaker {
private:
    SpreadTrendConfig* m_cfg;
    real64 m_std;
    real64 m_curCuScore;
    std::vector<real64> m_cuScores;
    std::vector<real64> m_slopes;
public:
    SpreadTrend(MinbarPairTrader* p) : DecisionMaker(p) {
        m_cfg = new SpreadTrendConfig;
    }
    ~SpreadTrend();


    void stats();
    void init();

    FXAct getDecision();
    bool isContinue();
};
