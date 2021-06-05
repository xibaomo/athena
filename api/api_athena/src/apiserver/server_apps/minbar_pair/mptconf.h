/*
 * =====================================================================================
 *
 *       Filename:  mptconf.h
 *
 *    Description:  Configs for pair trader
 *
 *        Version:  1.0
 *        Created:  05/26/2019 04:41:32 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MINBAR_PAIR_TRADER_CONFIG_H_
#define  _MINBAR_PAIR_TRADER_CONFIG_H_

#include "basics/baseconf.h"
const String MPT_ROOT = "MINBAR_PAIR_TRADER/";

class MptConfig : public BaseConfig {
private:
    MptConfig() {;}
public:
    virtual ~MptConfig() {;}

    static MptConfig& getInstance() {
        static MptConfig _ins;
        return _ins;
    }

    String getPairSymX() {
        return getKeyValue<String>(MPT_ROOT + "PAIR_SYM_X");
    }

    String getPairSymY() {
        return getKeyValue<String>(MPT_ROOT + "PAIR_SYM_Y");
    }

    int getLRLen() {
        return getKeyValue<int>(MPT_ROOT + "LR_LEN");
    }

    real32 getLowThresholdStd() {
        return getKeyValue<real32>(MPT_ROOT + "LOW_THRESHOLD_STD");
    }

    real32 getHighThresholdStd() {
        return getKeyValue<real32>(MPT_ROOT + "HIGH_THRESHOLD_STD");
    }

    real32 getCorrBaseline() {
        return getKeyValue<real32>(MPT_ROOT + "CORR_BSL");
    }

    real32 getR2Baseline() {
        return getKeyValue<real32>(MPT_ROOT + "R2_BSL");
    }

    int getStationaryCheckLookback() {
        return getKeyValue<int>(MPT_ROOT + "STATIONARY_CHECK_LOOKBACK");
    }

    real32 getStationaryPVLimit() {
        return getKeyValue<real32>(MPT_ROOT + "STATIONARY_PV_LIMIT");
    }

};
#endif   /* ----- #ifndef _MINBAR_PAIR_TRADER_CONFIG_H_  ----- */
