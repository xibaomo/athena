/*
 * =====================================================================================
 *
 *       Filename:  mean_reverse.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/03/2021 02:02:30 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once

#include "decision_maker.h"

class MeanReverse : public DecisionMaker {
private:
    real64 m_curMean;
    real64 m_orgMean;
    std::vector<real64> m_meanShift;

    real64 m_curStd;

    real64 m_minDev;
    real64 m_maxDev;

    std::vector<real64> m_devs;
public:
    MeanReverse(MinbarPairTrader* trader) : DecisionMaker(trader) {;}
    ~MeanReverse();

    void init();
    void stats();
    FXAct getDecision();

    bool isContinue();
};
