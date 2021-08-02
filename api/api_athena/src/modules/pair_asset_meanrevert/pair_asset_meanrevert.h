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

class PairAssetMeanRevert : public DecisionMaker {
  private:
    enum PairPosDirection {
        SAME,
        OPPOSITE
    };
    enum StationaryState {
        YES,
        NO
    };
    struct DevInfo {
        real64 buy;
        real64 sell;
    };
    real64 m_lotx, m_loty;
    real64 m_curMean;

    real64 m_devUnit; // std of past m_lookback

    std::vector<DevInfo> m_devs;

    bool m_isModelValid;

    // a*y+b*x (a>0) is defined as the total asset
    // if b > 0, buy of total asset uses y_buy,x_buy, sell uses y_sell, x_sell
    // if b < 0, buy of total asset uses y_buy,x_sell, sell uses y_sell, x_buy
    std::vector<real64> m_totalAssets_buy;
    std::vector<real64> m_totalAssets_sell;
    std::vector<real64> m_totalAssets_mid;

    size_t m_buys,m_sells;

    real64  m_highBuyDev, m_lowBuyDev;
    real64  m_highSellDev, m_lowSellDev;

    PairPosDirection m_pairPosDir;
    StationaryState  m_stationaryState;
  public:
    PairAssetMeanRevert(MinbarPairTrader* p) : DecisionMaker(p), m_isModelValid(true),m_buys(0),m_sells(0) {
        m_highBuyDev = m_lowBuyDev = m_highSellDev = m_lowSellDev = 0.f;
    }
    ~PairAssetMeanRevert();
    void init() override;
    void createTotalAssets();
    /**
     * Find best lot_x/y which is closest to the fitted slope
     */
    void findBestLots(real64 lotx0, real64 loty0);

    FXAct getDecision() override;
    void appendAssets();
    void checkStationaryState();

    FXAct decision_stationary();
    FXAct decision_non_stationary();

    real64 compDevFromMean();

    bool isContinue() override;

    void getBestLots(real64& lx, real64& ly) override {
        lx = m_lotx;
        ly = m_loty;
    }
};
