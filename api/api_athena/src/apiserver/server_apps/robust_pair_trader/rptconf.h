/*
 * =====================================================================================
 *
 *       Filename:  rptconf.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/10/2019 01:38:19 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _ROBUST_PAIRTRADER_CONF_H_
#define  _ROBUST_PAIRTRADER_CONF_H_

#include "basics/baseconf.h"
const String RPT_ROOT = "ROBUST_PAIR_TRADER/";

class RptConfig : public BaseConfig {
private:
    RptConfig(){;}
public:
    static RptConfig& getInstance() {
        static RptConfig _ins;
        return _ins;
    }

    String getSymX() {
        return getKeyValue<String>(RPT_ROOT + "SYM_X");
    }

    String getSymY() {
        return getKeyValue<String>(RPT_ROOT + "SYM_Y");
    }

    std::vector<real32> getThresholdStd() {
        return getKeyValue<std::vector<real32>>(RPT_ROOT + "THRESHOLD_STD");
    }

    real32 getOutlierWeightRatio() {
        return getKeyValue<real32>(RPT_ROOT + "OUTLIER_WEIGHT_RATIO");
    }

    int getOutlierNumLimit() {
        return getKeyValue<int>(RPT_ROOT + "OUTLIER_NUM_LIMIT");
    }

    real32 getStationaryPVLimit() {
        return getKeyValue<real32>(RPT_ROOT + "STATIONARY_PV_LIMIT");
    }

    real32 getR2Limit() {
        return getKeyValue<real32>(RPT_ROOT + "R2_LIMIT");
    }
};
#endif   /* ----- #ifndef _ROBUST_PAIRTRADER_CONF_H_  ----- */
