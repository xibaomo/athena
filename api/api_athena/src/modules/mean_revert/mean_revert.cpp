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
using namespace std;
using namespace athena;

MeanRevert::~MeanRevert() {
    //dumpVectors("devs.csv",m_buyYDevs, m_sellYDevs);
    dumpDevs("devs.csv");
    ostringstream oss;
    oss << "Num buys: " << m_buys << ", sells: " << m_sells << ", close_all: " << m_numclose;
    Log(LOG_INFO) << oss.str();
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
MeanRevert::init() {
    auto& spreads = m_trader->getSpreads();

    real64 mean = gsl_stats_mean(&spreads[0], 1, spreads.size());

    real64 s = gsl_stats_sd_m(&spreads[0], 1, spreads.size(), mean);
    Log(LOG_INFO) << "std of spreads: " + to_string(s);

    m_devUnit = findMedianDev(spreads, mean);
    Log(LOG_INFO) << "median deviation of spreads from its mean: " + to_string(m_devUnit);
}

void
MeanRevert::stats() {
    SpreadInfo lsp = m_trader->getLatestSpread();
    real64 ma = compLatestSpreadMA();
    SpreadInfo dev;
    dev.buy = (lsp.buy-ma) / m_devUnit;
    dev.sell = (lsp.sell-ma) / m_devUnit;
    m_spreadDevs.push_back(dev);

    ostringstream oss;
    oss << "spread dev/devUnit: buy: " << dev.buy << ", sell: " << dev.sell;
    Log(LOG_INFO) << oss.str();
}

FXAct
MeanRevert::getDecision() {
    stats();
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
    auto& spreads = m_trader->getSpreads();
    int len = m_trader->getConfig()->getSpreadMALookback();

    int start = spreads.size() - len;
    start = start < 0 ? 0 : start;
    real64 sum = std::accumulate(spreads.begin()+start, spreads.end(), 0.f);
    return sum / len;
}
