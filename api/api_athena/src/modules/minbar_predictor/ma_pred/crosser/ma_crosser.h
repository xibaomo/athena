/*
 * =====================================================================================
 *
 *       Filename:  ma_crosser.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  05/12/2019 05:05:02 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MA_CROSSER_PRED_H_
#define  _MA_CROSSER_PRED_H_

#include "minbar_predictor/ma_pred/ma_pred_base/ma_pred_base.h"
#include "minbar_predictor/ma_pred/ma_cal/ma_cal.h"
#include "mcconf.h"

class MACrosser : public MABasePredictor {
    struct ActionRecord {
        real32 long_ma;
        real32 short_ma;
        FXAction action;
        ActionRecord(real32 lma, real32 sma, FXAction ac) :
            long_ma(lma),short_ma(sma),action(ac){;}
    };

protected:
    std::vector<real32> m_median;
    std::vector<real32> m_long_ma;
    std::vector<real32> m_short_ma;

    std::vector<ActionRecord> m_records;

    MACrosserConfig* m_config;

    MACrosser(const String& cf, MACalculator* cal);
public:
    virtual ~MACrosser();

    static MACrosser& getInstance(const String& cf, MACalculator* cal) {
        static MACrosser _ins(cf, cal);
        return _ins;
    }

    void dumpRecords();
    void prepare();

    FXAction predict();

    int findNearestCross();
};
#endif   /* ----- #ifndef _MA_CROSSER_PRED_H_  ----- */
