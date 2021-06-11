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

class MeanRevert : public DecisionMaker {
private:
    real64 m_std;
    std::vector<real64> m_devs;

    size_t m_buys,m_sells,m_numclose;
public:
    MeanRevert(MinbarPairTrader* p) : DecisionMaker(p),m_buys(0),m_sells(0),m_numclose(0) {;}
    ~MeanRevert();
    void init();

    void stats();
    FXAct getDecision();

    bool isContinue() { return true; }
};
