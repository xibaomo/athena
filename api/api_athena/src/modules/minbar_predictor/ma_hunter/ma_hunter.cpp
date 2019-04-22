/*
 * =====================================================================================
 *
 *       Filename:  ma_hunter.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/18/2019 03:18:00 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "ma_hunter.h"

using namespace std;
using namespace athena;

MAHunter::MAHunter(const String& cfg) : m_maCal(nullptr),m_config(nullptr)
{
    m_config = &MahuntConfig::getInstance();
    m_config->loadConfig(cfg);

    m_maCal = createMACalculator(m_config->getMAType());
}

void
MAHunter::prepare()
{
    // Compute median
    m_median.resize(m_allMinBars->size());
    for(size_t i = 0; i < m_allMinBars->size(); i++) {
        auto& mb = (*m_allMinBars)[i];
        m_median[i] = (mb.high + mb.low) *.5;
    }

    m_maCal->compAllMA(m_median,m_config->getMALookback(),m_ma);
}

FXAction
MAHunter::predict()
{
    auto& mb = m_allMinBars->back();
    real32 md = (mb.high + mb.low)*.5;
    m_median.push_back(md);

    real32 ma = m_maCal->compLatestMA(m_median,m_config->getMALookback(),m_median.size()-1);
    m_ma.push_back(ma);
    return FXAction::NOACTION;
}
