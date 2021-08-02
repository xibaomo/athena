/*
 * =====================================================================================
 *
 *       Filename:  decision_maker.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  05/30/2021 05:02:43 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "fx_action/fx_action.h"

class MinbarPairTrader;
class DecisionMaker {
  protected:
    int m_lookback;
    MinbarPairTrader* m_trader;

  public:
    DecisionMaker(MinbarPairTrader* p) : m_lookback(-1),m_trader(p) {;}

    virtual void init() = 0;

    virtual ~DecisionMaker() {;}

    virtual real64 getSlope() { return 0.f; }

    virtual void getBestLots(real64& lotx, real64& loty) {lotx=0.f;loty=0.f;}

    virtual FXAct getDecision() = 0;

    virtual bool isContinue() = 0;

    virtual void finish(){;}

    void setLookback(int lookback) { m_lookback = lookback; }
};
