/*
 * =====================================================================================
 *
 *       Filename:  mcconf.h
 *
 *    Description
 *
 *        Version:  1.0
 *        Created:  05/12/2019 05:14:24 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MA_CROSSER_CONFIG_H_
#define  _MA_CROSSER_CONFIG_H_

#include "basics/baseconf.h"
#include "minbar_predictor/ma_pred/ma_cal/ma_cal.h"

const String MA_CROSSER_ROOT = "MA_PREDICTOR/MA_CROSSER/";

class MACrosserConfig : public BaseConfig {
protected:
    MACrosserConfig() {;}
public:
    virtual ~MACrosserConfig(){;}

    static MACrosserConfig& getInstance() {
        static MACrosserConfig _ins;
        return _ins;
    }

    int getLongLookBack() {
        return getKeyValue<int>(MA_CROSSER_ROOT + "LONG_LOOKBACK");
    }

    int getShortLookBack() {
        return getKeyValue<int>(MA_CROSSER_ROOT + "SHORT_LOOKBACK");
    }

    int getTurnPointExpiry() {
        return getKeyValue<int>(MA_CROSSER_ROOT + "TURN_POINT_EXPIRY");
    }

    real64 getStartOffGap() {
        return getKeyValue<real64>(MA_CROSSER_ROOT + "STARTOFF_GAP");
    }
};
#endif   /* ----- #ifndef _MA_CROSSER_CONFIG_H_  ----- */
