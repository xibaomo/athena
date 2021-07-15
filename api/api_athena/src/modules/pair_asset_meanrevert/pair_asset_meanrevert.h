/*
 * =====================================================================================
 *
 *       Filename:  pair_asset_meanrevert.h
 *
 *    Description:  The paired symbols satisfy y = a*x + b + epsilon(x), equivalently, we are trading
 *                  a custom symbol (y - a*x), which is stationary.
 *                  In most cases, when lot size of y and/or x is quite small, the ratio of y and x hardly
 *                  ensures (y-a*x) is stationary.
 *                  This concrete oracle, will carefully select lot sizes of both x and y, so as to ensure
 *                  (lot_y*y +/- lot_x*x) is stationary, and the rest is still classical mean-reverting algorithm.
 *
 *
 * =====================================================================================
 */
#pragma once

#include "minbar_pair_trader/decision_maker.h"
#include "minbar_pair_trader/minbar_pair_trader.h"

class PairAssetMeanRevert : public DecisionMaker{
public:
    PairAssetMeanRevert(MinbarPairTrader* p) : DecisionMaker(p) {}

    void init() override;
    /**
     * Find best lot_x/y which is closest to the fitted slope
     */
    void findBestLots();

    FXAct getDecision() override;

    bool isContinue() override;
};
