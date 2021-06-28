/*
 * =====================================================================================
 *
 *       Filename:  dm_rule_75.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  05/30/2021 05:06:27 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#if 0
#include "dm_rule_75.h"
#include "basics/utils.h"
#include "minbar_pair_trader.h"
#include <gsl/gsl_statistics.h>
using namespace std;
using namespace athena;

Rule75::~Rule75() {
    dumpVectors("devs.csv",m_deviations);

    Log(LOG_INFO) << "Num buys: " + to_string(m_numBuys) + ", num sells: " + to_string(m_numSells);
}

void
Rule75::init() {
    auto& spreads = m_trader->getSpreads();

    m_median = gsl_stats_mean(&spreads[0],1,spreads.size());
    Log(LOG_INFO) << "Mean of spreads: " + to_string(m_median);

    m_std = gsl_stats_sd_m(&spreads[0],1,spreads.size(),m_median);
    Log(LOG_INFO) << "std of spreads: " + to_string(m_std);

//    m_median = gsl_stats_median(&spreads[0],1,spreads.size());
//    Log(LOG_INFO) << "Median of spreads: " + to_string(m_median);
}
void
Rule75::stats() {
    auto& spreads = m_trader->getSpreads();
    MptConfig* cfg = m_trader->getConfig();
    int len = cfg->getStationaryCheckLookback();

    size_t start = spreads.size() - len;
    real64 mean = gsl_stats_mean(&spreads[start],1,len);

    real64 mean_shift = (mean-m_median) / m_std;
    Log(LOG_INFO) << "Mean shift (in std) of latest " + to_string(len) + " spreads: " + to_string(mean_shift);
}
FXAct
Rule75::getDecision()
{
    stats();
    auto& spreads = m_trader->getSpreads();

    real64 cur = spreads.back();
    real64 dev = (cur-m_median)/m_std;
    Log(LOG_INFO) << "Latest error (std) relative to original mean: " + to_string(dev);
    m_deviations.push_back(dev);

    if(cur > m_median) {
        Log(LOG_INFO) << "Current error higher than median";
        m_numSells++;
        return FXAct::PLACE_SELL;
    } else if (cur < m_median) {
        Log(LOG_INFO) << "Current error lower than median";
        m_numBuys++;
        return FXAct::PLACE_BUY;
    } else {
    ;}
    return FXAct::NOACTION;
}

bool
Rule75::isContinue() {

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

void
Rule75::finish() {
    cout<<"rule 75 finish called"<<endl;
}
#endif // 0
