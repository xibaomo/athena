/*
 * =====================================================================================
 *
 *       Filename:  ma_cal.h
 *
 *    Description:  Base of MA calculator
 *
 *        Version:  1.0
 *        Created:  04/22/2019 02:11:40 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BASE_PREDICT_MA_CALCULATOR_H_
#define  _BASE_PREDICT_MA_CALCULATOR_H_

#include <vector>
#include "minbar_predictor/mb_base/mb_base_pred.h"

enum class MA_TYPE {
    LWMA,
    EMA,
    SMA
};

class MACalculator {
protected:

public:
    virtual ~MACalculator() {;}

    virtual void compAllMA(std::vector<real32>& arr, int lookback,std::vector<real32>& out) = 0;
    virtual real32 compLatestMA(std::vector<real32>& arr, int lookback, size_t idx) = 0;
};

MACalculator* createMACalculator(MA_TYPE mt);
#endif   /* ----- #ifndef _BASE_PREDICT_MA_CALCULATOR_H_  ----- */
