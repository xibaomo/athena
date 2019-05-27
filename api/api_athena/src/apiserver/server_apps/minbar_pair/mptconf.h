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

    String getPairASym() {
        return getKeyValue<String>(MPT_ROOT + "PAIR_A_SYM");
    }

    String getPairBSym() {
        return getKeyValue<String>(MPT_ROOT + "PAIR_B_SYM");
    }

    int getLRLen() {
        return getKeyValue<int>(MPT_ROOT + "LR_LEN");
    }

    real32 getThresholdStd() {
        return getKeyValue<real32>(MPT_ROOT + "THRESHOLD_STD");
    }
};
#endif   /* ----- #ifndef _MINBAR_PAIR_TRADER_CONFIG_H_  ----- */
