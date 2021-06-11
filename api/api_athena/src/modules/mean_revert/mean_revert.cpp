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
    dumpVectors("devs.csv",m_devs);
    ostringstream oss;
    oss << "Num buys: " << m_buys << ", sells: " << m_sells << ", close_all: " << m_numclose;
    Log(LOG_INFO) << oss.str();
}

void
MeanRevert::init() {
    auto& spreads = m_trader->getSpreads();

    real64 mean = gsl_stats_mean(&spreads[0],1,spreads.size());

    m_std = gsl_stats_sd_m(&spreads[0],1,spreads.size(),mean);
    Log(LOG_INFO) << "std of spreads: " + to_string(m_std);
}

void
MeanRevert::stats() {
    auto& spreads = m_trader->getSpreads();
    real64 dev = spreads.back()/m_std;
    m_devs.push_back(dev);

    Log(LOG_INFO) << "Current dev/std: " + to_string(dev);
}

FXAct
MeanRevert::getDecision() {
    stats();
    MptConfig* cfg = m_trader->getConfig();

    real64 low_thd = cfg->getLowThresholdStd();
    real64 high_thd = cfg->getHighThresholdStd();

    real64 dev = m_devs.back();
    if (dev >= high_thd) {
        m_sells++;
        return FXAct::PLACE_SELL;
    }

    if (dev <= -high_thd) {
        m_buys++;
        return FXAct::PLACE_BUY;
    }

    if (abs(dev) <= low_thd) {
        m_numclose++;
        return FXAct::CLOSE_ALL_POS;
    }

    real64 old_dev = m_devs[m_devs.size()-2];
    if(old_dev * dev < 0) {
        m_numclose++;
        return FXAct::CLOSE_ALL_POS;
    }

    return FXAct::NOACTION;
}
