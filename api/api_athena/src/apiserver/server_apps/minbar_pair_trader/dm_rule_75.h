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
class Rule75 : public DecisionMaker {
public:

    Rule75(MinbarPairTrader* trader) : DecisionMaker(trader){;}
    FXAct getDecision();
    bool isContinue();
};
