/*
 * =====================================================================================
 *
 *       Filename:  mpconf.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/24/2019 12:17:32 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MINBAR_PREDICTOR_CONFIG_H_
#define  _MINBAR_PREDICTOR_CONFIG_H_

#include "basics/baseconf.h"
#include "minbar_predictor/ma_pred/ma_cal/ma_cal.h"

enum class MP_Type : int{
    MA_GRAD_HUNT = 0,
    LS_MA_CROSS, // long-short MA cross
};

const String MP_ROOT = "MA_PREDICTOR/";

class MPConfig : public BaseConfig {
protected:
    std::unordered_map<String, MA_TYPE> m_str2matype;
    MPConfig() {
        m_str2matype["LWMA"] = MA_TYPE::LWMA;
        m_str2matype["EMA"]  = MA_TYPE::EMA;
        m_str2matype["SMA"]  = MA_TYPE::SMA;
    }
public:
    virtual ~MPConfig() {;}
    static MPConfig& getInstance() {
        static MPConfig _ins;
        return _ins;
    }

    MP_Type getMAPredictorType() {
        return (MP_Type)getKeyValue<int>(MP_ROOT + "MA_PREDICTOR_TYPE");
    }

    MA_TYPE getMAType() {
        String tmp = getKeyValue<String>(MP_ROOT + "MA_TYPE");
        return m_str2matype[tmp];
    }
};
#endif   /* ----- #ifndef _MINBAR_PREDICTOR_CONFIG_H_  ----- */
