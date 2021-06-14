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
    //dumpVectors("devs.csv",m_buyYDevs,m_sellYDevs);
    dumpDevs("devs.csv");
    ostringstream oss;
    oss << "Num buys: " << m_buys << ", sells: " << m_sells << ", close_all: " << m_numclose;
    Log(LOG_INFO) << oss.str();
}

real64
MeanRevert::findMedianDev(const std::vector<real64>& spreads, const real64 mean) {
    std::vector<real64> devs(spreads.size());

    for(size_t i=0; i < spreads.size(); i++) {
        devs[i] = abs(spreads[i]-mean);
    }

    std::sort(devs.begin(),devs.end());

    return devs[devs.size()/2-1];
}

void
MeanRevert::init() {
    auto& spreads = m_trader->getSpreads();

    real64 mean = gsl_stats_mean(&spreads[0],1,spreads.size());

    real64 s = gsl_stats_sd_m(&spreads[0],1,spreads.size(),mean);
    Log(LOG_INFO) << "std of spreads: " + to_string(s);
    m_devUnit = findMedianDev(spreads,mean);
    Log(LOG_INFO) << "median deviation of spreads from its mean: " + to_string(m_devUnit);
}

void
MeanRevert::stats() {
    SpreadInfo buy_y_spread = m_trader->getLatestBuyYSpread();
    SpreadInfo dev;
    dev.create = buy_y_spread.create / m_devUnit;
    dev.close  = buy_y_spread.close / m_devUnit;
    m_buyYDevs.push_back(dev);

    Log(LOG_INFO) << "buy_y_spread  dev / devUnit: " + to_string(dev.create) + "," + to_string(dev.close);

    SpreadInfo sell_y_spread = m_trader->getLatestSellYSpread();
    dev.create = sell_y_spread.create / m_devUnit;
    dev.close  = sell_y_spread.close  / m_devUnit;
    m_sellYDevs.push_back(dev);

    Log(LOG_INFO) << "sell_y_spread dev / devUnit: " + to_string(dev.create) + "," + to_string(dev.close);
}

FXAct
MeanRevert::getDecision() {
    stats();
    MptConfig* cfg = m_trader->getConfig();

    real64 low_thd = cfg->getLowThresholdStd();
    real64 high_thd = cfg->getHighThresholdStd();

    if (m_buyYDevs.back().create <= -high_thd) {
        m_buys++;
        return FXAct::PLACE_BUY;
    }

    if (m_sellYDevs.back().create >= high_thd) {
        m_sells++;
        return FXAct::PLACE_SELL;
    }

    if (abs(m_buyYDevs.back().close) <= low_thd) {
        m_numclose++;
        return FXAct::CLOSE_BUY;
    }

    if(abs(m_sellYDevs.back().close) <= low_thd) {
        m_numclose++;
        return FXAct::CLOSE_SELL;
    }

    real64 old_dev = m_buyYDevs[m_buyYDevs.size()-2].close;
    real64 new_dev = m_buyYDevs.back().close;
    if(old_dev * new_dev < 0) {
        m_numclose++;
        return FXAct::CLOSE_BUY;
    }

    old_dev = m_sellYDevs[m_sellYDevs.size()-2].close;
    new_dev = m_sellYDevs.back().close;
    if(old_dev * new_dev < 0) {
        m_numclose++;
        return FXAct::CLOSE_SELL;
    }

    return FXAct::NOACTION;
}

void
MeanRevert::dumpDevs(const String& fn) {
    std::vector<real64> buy_y_spread_create;
    std::vector<real64> buy_y_spread_close;
    std::vector<real64> sell_y_spread_create;
    std::vector<real64> sell_y_spread_close;
    for(size_t i=0; i < m_buyYDevs.size();i++) {
        buy_y_spread_create.push_back(m_buyYDevs[i].create);
        buy_y_spread_close.push_back(m_buyYDevs[i].close);
        sell_y_spread_create.push_back(m_sellYDevs[i].create);
        sell_y_spread_close.push_back(m_sellYDevs[i].close);
    }
    dumpVectors(fn,buy_y_spread_create,buy_y_spread_close,sell_y_spread_create,sell_y_spread_close);
}
