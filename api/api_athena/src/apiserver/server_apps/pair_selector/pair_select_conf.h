/*
 * =====================================================================================
 *
 *       Filename:  mulpairconf.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  06/16/2019 07:41:19 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MUL_PAIR_TRADER_H_
#define  _MUL_PAIR_TRADER_H_

#include "basics/baseconf.h"
const String MUPAIR_ROOT = "MULTIPAIR_TRADER/";
class PairSelectConfig : public BaseConfig {
private:
    PairSelectConfig() {;}
public:
    virtual ~PairSelectConfig() {;}

    static PairSelectConfig& getInstance() {
        static PairSelectConfig _ins;
        return _ins;
    }

    real32 getCorrBaseline() {
        return getKeyValue<real32>(MUPAIR_ROOT + "CORR_BSL");
    }
};

#endif   /* ----- #ifndef _MUL_PAIR_TRADER_H_  ----- */
