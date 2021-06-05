/*
 * =====================================================================================
 *
 *       Filename:  spread_trend.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/04/2021 05:08:01 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "spread_trend.h"
#include "minbar_pair_trader/minbar_pair_trader.h"
#include "basics/utils.h"
using namespace std;
using namespace athena;

SpreadTrend::~SpreadTrend() {
    dumpVectors("cuscores.csv",m_cuScores);
}
void
SpreadTrend::init() {
    auto& spreads = m_trader->getSpreads();

    real64 mean = gsl_stats_mean(&spreads[0],1,spreads.size());
    Log(LOG_INFO) << "Mean of spreads: " + to_string(mean);

    m_std = gsl_stats_sd_m(&spreads[0],1,spreads.size(),mean);
    Log(LOG_INFO) << "std of spreads: " + to_string(m_std);

    m_curCuScore = std::accumulate(spreads.begin(),spreads.end(),0.f) / m_std;
    Log(LOG_INFO) << "Base cuScore (std): " + to_string(m_curCuScore);

    m_cuScores.push_back(m_curCuScore);
}

FXAct
SpreadTrend::getDecision() {
    auto& spreads = m_trader->getSpreads();
    m_curCuScore += spreads.back() / m_std;
    Log(LOG_INFO) << "Current cuScore (std): " + to_string(m_curCuScore);

    m_cuScores.push_back(m_curCuScore);
    return FXAct::NOACTION;
}

bool
SpreadTrend::isContinue() {
    return true;
}
