/*
 * =====================================================================================
 *
 *       Filename:  stconf.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/05/2021 12:37:43 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once
#include "conf/generalconf.h"

const String SPREAD_TREND_ROOT = "SPREAD_TREND/";
class SpreadTrendConfig {
    GeneralConfig* m_cfg;
public:
    SpreadTrendConfig() {
        m_cfg = &GeneralConfig::getInstance();
    }

    int getSlopeLookback() {
        return m_cfg->getKeyValue<int>(SPREAD_TREND_ROOT + "SLOPE_LOOKBACK");
    }

    real64 getSlopeThreshold() {
        return m_cfg->getKeyValue<real64>(SPREAD_TREND_ROOT + "SLOPE_THRESHOLD");
    }
};
