/*
 * =====================================================================================
 *
 *       Filename:  dm_rule_75.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  05/30/2021 05:05:31 AM
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
#include <string>
class Rule75 : public DecisionMaker {
private:
    real64 m_median;
    real64 m_std;

    std::vector<real64> m_deviations; // in unit of m_std
    size_t m_numBuys;
    size_t m_numSells;
public:

    Rule75(MinbarPairTrader* trader) : DecisionMaker(trader), m_numBuys(0),m_numSells(0){;}

    ~Rule75();

    void init();
    void stats();
    FXAct getDecision();
    bool isContinue();

    void finish();
};
