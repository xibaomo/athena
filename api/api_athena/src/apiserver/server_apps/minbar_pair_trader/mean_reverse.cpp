/*
 * =====================================================================================
 *
 *       Filename:  mean_reverse.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/03/2021 02:03:19 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#if 0
#include "mean_reverse.h"
#include "basics/utils.h"
#include "minbar_pair_trader.h"
#include <gsl/gsl_statistics.h>
using namespace std;
using namespace athena;
MeanReverse::~MeanReverse() {
    dumpVectors("devs.csv",m_devs);
    dumpVectors("mean_shift.csv",m_meanShift);
    Log(LOG_INFO) << "Min dev/std: " + to_string(m_minDev) + ", max dev/std: " + to_string(m_maxDev);
}

void
MeanReverse::init() {
    auto& spreads = m_trader->getSpreads();

    m_curMean = gsl_stats_mean(&spreads[0],1,spreads.size());
    Log(LOG_INFO) << "Mean of spreads: " + to_string(m_curMean);
    m_orgMean = m_curMean;

    m_curStd = gsl_stats_sd_m(&spreads[0],1,spreads.size(),m_curMean);
    Log(LOG_INFO) << "std of spreads: " + to_string(m_curStd);
}

void
MeanReverse::stats() {
    auto& spreads = m_trader->getSpreads();
    MptConfig* cfg = m_trader->getConfig();
    int len = cfg->getStationaryCheckLookback();

    size_t start = spreads.size() - len;
    m_curMean = gsl_stats_mean(&spreads[start],1,len-1); // rule out the latest spread

    real64 ms = (m_curMean-m_orgMean)/m_curStd;
    Log(LOG_INFO) << "Mean shift (std) w.r.t original mean: " + to_string(ms);
    m_meanShift.push_back(ms);

    real64 dev = (spreads.back() - m_curMean)/m_curStd;

    m_devs.push_back(dev);

    if(abs(dev) > m_maxDev)
        m_maxDev = dev;
    if(abs(dev) < m_minDev)
        m_minDev = dev;

    Log(LOG_INFO) << "dev/std: " + to_string(dev);
}

FXAct
MeanReverse::getDecision() {
    stats();

    MptConfig* cfg = m_trader->getConfig();

    real64 low_thd = cfg->getLowThresholdStd();
    real64 high_thd = cfg->getHighThresholdStd();

    real64 dev = m_devs.back();

    if(dev >= high_thd) {
        return FXAct::PLACE_SELL;
    } else if (dev <= -high_thd) {
        return FXAct::PLACE_BUY;
    } else if (fabs(dev) <= low_thd){
        return FXAct::CLOSE_ALL_POS;
    } else {
        return FXAct::NOACTION;
    }

    return FXAct::NOACTION;
}

bool
MeanReverse::isContinue() {
    MptConfig* cfg = m_trader->getConfig();
    int len = cfg->getStationaryCheckLookback();

    auto& spreads = m_trader->getSpreads();
    size_t start = spreads.size() - len;
    real64 pv = testADF(&spreads[start],len);

    Log(LOG_INFO) << "p-value of ADF test of latest " + to_string(len) + " spreads: " + to_string(pv);

    if(pv > cfg->getStationaryPVLimit()) {
        Log(LOG_WARNING) << "Spreads become non-stationary. Model fails!";
        return false;
    }

    return true;
}
#endif // 0
