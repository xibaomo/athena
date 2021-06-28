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
#if 0
#include "spread_trend.h"
#include "minbar_pair_trader/minbar_pair_trader.h"
#include "basics/utils.h"
#include "linreg/linreg.h"
#include "linreg/roblinreg.h"
using namespace std;
using namespace athena;

SpreadTrend::~SpreadTrend() {
    dumpVectors("cuscores.csv",m_cuScores);
    dumpVectors("slopes.csv",m_slopes);

    if(m_cfg) delete m_cfg;
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

void
SpreadTrend::stats() {
    auto& spreads = m_trader->getSpreads();
    m_curCuScore += spreads.back() / m_std;
    Log(LOG_INFO) << "Current cuScore (std): " + to_string(m_curCuScore);

    m_cuScores.push_back(m_curCuScore);

    auto& x = m_trader->getOpenX();
    auto& y = m_trader->getOpenY();

    size_t len = 500;
    size_t start = x.size() - len;
    //LRParam param = linreg(&x[start],&y[start],len);
    RobLRParam param = robLinreg(&x[start],&y[start],len);
    m_slopes.push_back(param.c1);
}
FXAct
SpreadTrend::getDecision() {
    stats();

    int len = m_cfg->getSlopeLookback();
    real64 slope_thd = m_cfg->getSlopeThreshold();

    vector<real64> x(len);
    iota(x.begin(),x.end(),0.f);
    size_t s = m_cuScores.size()-len;
    LRParam param = linreg(&x[0],&m_cuScores[s],len);

    Log(LOG_INFO) << "cuScore trend slope: " + to_string(param.c1);

//    if (param.c1 > slope_thd) {
//        return FXAct::PLACE_BUY;
//    }
//
//    if (param.c1 < -slope_thd) {
//        return FXAct::PLACE_SELL;
//    }

    return FXAct::NOACTION;
}

bool
SpreadTrend::isContinue() {
    return true;
}
#endif // 0
