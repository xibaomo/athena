/*
 * =====================================================================================
 *
 *       Filename:  mhconf.h
 *
 *    Description:  This file defines MA hunter config
 *
 *        Version:  1.0
 *        Created:  04/18/2019 02:35:52 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MA_HUNTER_CONFIG_H_
#define  _MA_HUNTER_CONFIG_H_

#include "basics/baseconf.h"
#include "minbar_predictor/ma_pred/ma_cal/ma_cal.h"

const String MA_HUNTER_ROOT = "MA_PREDICTOR/MA_GRAD_HUNTER/";


class MahuntConfig : public BaseConfig {
protected:
    std::unordered_map<String,MA_TYPE> m_str2matype;
    MahuntConfig() {
        m_str2matype["LWMA"] = MA_TYPE::LWMA;
        m_str2matype["EMA"]  = MA_TYPE::EMA;
        m_str2matype["SMA"]  = MA_TYPE::SMA;
    }
public:
    virtual ~MahuntConfig() {;}
    static MahuntConfig& getInstance() {
        static MahuntConfig _ins;
        return _ins;
    }

    int getMALookback() {
        return getKeyValue<int>(MA_HUNTER_ROOT + "LOOKBACK");
    }

    MA_TYPE getMAType() {
        String tmp = getKeyValue<String>(MA_HUNTER_ROOT + "MA_TYPE");
        return m_str2matype[tmp];
    }

    int getTurnPointOffset() {
        return getKeyValue<int>(MA_HUNTER_ROOT + "TURN_POINT_OFFSET");
    }

    real32 getSlopeZero() {
        return getKeyValue<real32>(MA_HUNTER_ROOT + "SLOPE_ZERO");
    }

    real64 getFireOffSlope() {
        return getKeyValue<real64>(MA_HUNTER_ROOT + "STARTOFF_SLOPE");
    }

};
#endif   /* ----- #ifndef _MA_HUNTER_CONFIG_H_  ----- */
