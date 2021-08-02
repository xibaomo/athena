/*
 * =====================================================================================
 *
 *       Filename:  mean_revert.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/11/2021 11:25:11 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "mean_revert.h"
#include "minbar_pair_trader/minbar_pair_trader.h"
#include "basics/utils.h"
#include <sstream>
#include <numeric>
using namespace std;
using namespace athena;

MeanRevert::~MeanRevert() {
    dumpVectors("cuscore.csv",m_cuScores);
    dumpDevs("devs.csv");
    dumpTradeSpreads();
    dumpVectors("spreads.csv",m_spreads);
    ostringstream oss;
    oss << "Num buys: " << m_buys << ", sells: " << m_sells << ", close_all: " << m_numclose;
    Log(LOG_INFO) << oss.str();

    ostringstream os;
    os << "Dev range: buy: [" << m_lowBuyDev<<","<<m_highBuyDev<<"], sell: [" << m_lowSellDev << "," << m_highSellDev << "]";
    Log(LOG_INFO) << os.str();

    Log(LOG_INFO) << "Linear regression done. c0: " + to_string(m_linParam.c0) + ", c1: " + to_string(m_linParam.c1);

}

real64
MeanRevert::findMedianDev(const std::vector<real64>& spreads, const real64 mean) {
    std::vector<real64> devs(spreads.size());

    for ( size_t i = 0; i < spreads.size(); i++ ) {
        devs[i] = abs(spreads[i]-mean);
    }

    std::sort(devs.begin(), devs.end());

    return devs[devs.size()/2-1];
}

void
MeanRevert::updateModel(int len) {
    auto& vx = m_trader->getMidX();
    auto& vy = m_trader->getMidY();
    size_t start = vx.size() - len;
    real64* x = new real64[len];
    real64* y = new real64[len];

    real64 tsz_x = m_trader->getTickSizeX();
    real64 tsz_y = m_trader->getTickSizeY();
    real64 tv_x  = m_trader->getTickValX();
    real64 tv_y  = m_trader->getTickValY();

    auto& mid_x = m_trader->getMidX();
    auto& mid_y = m_trader->getMidY();
    int k=0;
    for ( size_t i = start; i < start+len; i++ ) {
        x[k] = mid_x[i] / tsz_x * tv_x;
        y[k++] = mid_y[i] / tsz_y * tv_y; // convert to dollars
    }

    m_linParam = linreg(x, y, len);
    //m_linParam = robLinreg(x, y, len);

    Log(LOG_INFO) << "Linear regression done. c0: " + to_string(m_linParam.c0) + ", c1: " + to_string(m_linParam.c1);

    real64 r2 = compR2(m_linParam, x, y, len);

    Log(LOG_INFO) << "Past " + to_string(len) + " pts R2: " + to_string(r2);

    delete[] x;
    delete[] y;
}
void
MeanRevert::compOldSpreads() {
    real64 ts_x = m_trader->getTickSizeX();
    real64 ts_y = m_trader->getTickSizeY();
    real64 tv_x  = m_trader->getTickValX();
    real64 tv_y  = m_trader->getTickValY();

    auto& mid_x = m_trader->getMidX();
    auto& mid_y = m_trader->getMidY();
    for ( size_t i = 0; i < mid_x.size(); i++ ) {
        real64 x = mid_x[i]/ts_x*tv_x;
        real64 y = mid_y[i]/ts_y*tv_y;
        real64 tmp = y - (m_linParam.c1 * x + m_linParam.c0);
        m_spreads.push_back(tmp);
    }

    real64 pv = testADF(&m_spreads[0], m_spreads.size());
    //real64 pv = 0.f;
    Log(LOG_INFO) << "p-value of stationarity of spreads: " + to_string(pv);
}

void
MeanRevert::init() {
    int len = m_trader->getMidX().size();
    updateModel(len);

    compOldSpreads();
    real64 mean = gsl_stats_mean(&m_spreads[0], 1, m_spreads.size());

    Log(LOG_INFO) << "Mean of spreads: " + to_string(mean);

    real64 s = gsl_stats_sd_m(&m_spreads[0], 1, m_spreads.size(), mean);
    Log(LOG_INFO) << "std of spreads: " + to_string(s);

    //m_devUnit = findMedianDev(spreads, mean);
    m_devUnit = 1.f;
    Log(LOG_INFO) << "median deviation (md) of spreads from its mean: " + to_string(m_devUnit);

    if(m_linParam.c1 > 0) {
        m_posPairDirection = OPPOSITE;
    } else {
        m_posPairDirection = SAME;
    }
}
//======================== Decision for each pair =============================
real64
MeanRevert::compSpread(real64 x, real64 y) {
    real64 err = y - (m_linParam.c0 + m_linParam.c1 * x);
    return err;
}
void
MeanRevert::compNewSpreads() {
    real64 ts_x = m_trader->getTickSizeX();
    real64 ts_y = m_trader->getTickSizeY();
    real64 tv_x = m_trader->getTickValX();
    real64 tv_y = m_trader->getTickValY();

    real64 x_ask = m_trader->getAskX().back();
    real64 y_ask = m_trader->getAskY().back();
    real64 x_bid = m_trader->getBidX().back();
    real64 y_bid = m_trader->getBidY().back();
    switch(m_posPairDirection) {
    case SAME: {
        SpreadInfo ts;
        ts.buy   = compSpread(x_ask/ts_x*tv_x, y_ask/ts_y*tv_y);
        ts.sell  = compSpread(x_bid/ts_x*tv_x, y_bid/ts_y*tv_y);

        m_tradeSpreads.push_back(ts);
        m_spreads.push_back((ts.buy+ts.sell)*.5f);
    }
    break;
    case OPPOSITE: {
        SpreadInfo ts;
        ts.buy  = compSpread(x_bid/ts_x*tv_x, y_ask/ts_y*tv_y);
        ts.sell = compSpread(x_ask/ts_x*tv_x, y_bid/ts_y*ts_y);

        m_tradeSpreads.push_back(ts);
        m_spreads.push_back((ts.buy+ts.sell)*.5f);
    }
    break;
    default:
        Log(LOG_FATAL) << "unknown position pair directions";
        break;
    }
}

void
MeanRevert::compDevFromMean() {
    SpreadInfo lsp = m_tradeSpreads.back();

    Log(LOG_INFO) << "Current mean: " + to_string(m_curMean);

    real64 ma = m_curMean;

    SpreadInfo dev;
    dev.buy = (lsp.buy-ma) / m_devUnit;
    dev.sell = (lsp.sell-ma) / m_devUnit;
    m_spreadDevs.push_back(dev);

    if (dev.buy > m_highBuyDev) m_highBuyDev = dev.buy;
    if (dev.buy < m_lowBuyDev)  m_lowBuyDev = dev.buy;
    if (dev.sell > m_highSellDev) m_highSellDev = dev.sell;
    if (dev.sell < m_lowSellDev)  m_lowSellDev = dev.sell;

    ostringstream os;
    os << "Spread buy: " << lsp.buy << ", sell: " << lsp.sell;
    Log(LOG_INFO) << os.str();

    ostringstream oss;
    oss << "spread dev/devUnit: buy: " << dev.buy << ", sell: " << dev.sell;
    Log(LOG_INFO) << oss.str();

}

FXAct
MeanRevert::getDecision() {
    updateModel(m_lookback);

    compNewSpreads();
    m_curMean = compLatestSpreadMean(m_lookback);

    compDevFromMean();

    MptConfig* cfg = m_trader->getConfig();

    real64 low_thd = cfg->getLowThresholdStd();
    real64 high_thd = cfg->getHighThresholdStd();

    if ( m_spreadDevs.back().buy <= -high_thd ) {
        m_buys++;
        return FXAct::PLACE_BUY;
    }

    if ( m_spreadDevs.back().sell >= high_thd ) {
        m_sells++;
        return FXAct::PLACE_SELL;
    }

    if ( abs(m_spreadDevs.back().sell) <= low_thd ) {
        m_numclose++;
        return FXAct::CLOSE_BUY;
    }

    if ( abs(m_spreadDevs.back().buy) <= low_thd ) {
        m_numclose++;
        return FXAct::CLOSE_SELL;
    }

    real64 old_dev = m_spreadDevs[m_spreadDevs.size()-2].sell;
    real64 new_dev = m_spreadDevs.back().sell;
    if ( old_dev * new_dev < 0 ) {
        m_numclose++;
        return FXAct::CLOSE_BUY;
    }

    old_dev = m_spreadDevs[m_spreadDevs.size()-2].buy;
    new_dev = m_spreadDevs.back().buy;
    if ( old_dev * new_dev < 0 ) {
        m_numclose++;
        return FXAct::CLOSE_SELL;
    }

    return FXAct::NOACTION;
}

void
MeanRevert::dumpDevs(const String& fn) {
    std::vector<real64> buy_dev;
    std::vector<real64> sell_dev;

    for ( size_t i = 0; i < m_spreadDevs.size(); i++ ) {
        buy_dev.push_back(m_spreadDevs[i].buy);
        sell_dev.push_back(m_spreadDevs[i].sell);
    }
    dumpVectors(fn, buy_dev, sell_dev);
}

real64
MeanRevert::compLatestSpreadMA() {
    auto& spreads = m_spreads;
    int len = m_trader->getConfig()->getSpreadMALookback();

    int start = spreads.size() - len;
    start = start < 0 ? 0 : start;
    real64 sum = std::accumulate(spreads.begin()+start, spreads.end(), 0.f);
    return sum / len;
}

real64
MeanRevert::compLatestSpreadMean(size_t len) {
    auto& asset_x = m_trader->getAssetX_mid();
    auto& asset_y = m_trader->getAssetY_mid();
    vector<real64> spreads(len);

    int start = asset_x.size() - len;
    for(int i=0; i< len; i++) {
        spreads[i] = asset_y[start+i] - (m_linParam.c0 + m_linParam.c1*asset_x[start+i]);
    }

    real64 s = std::accumulate(spreads.begin(),spreads.end(),0.f);

    return s/len;
}

void
MeanRevert::dumpTradeSpreads() {
    std::vector<real64> v1,v2;
    for(auto& p : m_tradeSpreads) {
        v1.push_back(p.buy);
        v2.push_back(p.sell);
    }
    dumpVectors("trade_spreads.csv",v1,v2);
}
