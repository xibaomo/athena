/*
 * =====================================================================================
 *
 *       Filename:  ma_lwma.h
 *
 *    Description:  Linear-weighted MA
 *
 *        Version:  1.0
 *        Created:  04/22/2019 02:13:33 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MA_LWMA_H_
#define  _MA_LWMA_H_

#include "ma_cal.h"
class MA_lwma : public MACalculator
{
protected:
    MA_lwma() {;}
public:
    virtual ~MA_lwma() {;}

    static MA_lwma& getInstance(){
        static MA_lwma _ins;
        return _ins;
    }

    real32 compLatestMA(std::vector<real32>& arr, int lookback, size_t i)
    {
        if (i < lookback-1) {
            return NAN;
        }

        real32 s = 0.f;
        int w = 1;
        float W = (1. + lookback) * lookback*0.5;
        for (int m = i - lookback + 1; m <= i; m++) {
            s += arr[m] * w++;
        }
        return s/W;
    }

    void compAllMA(std::vector<real32>& arr, int lookback, std::vector<real32>& out)
    {
        out.resize(arr.size());
        for (size_t i = 0; i < out.size(); i++) {
            out[i] = compLatestMA(arr,lookback,i);
        }
    }
};
#endif   /* ----- #ifndef _MA_LWMA_H_  ----- */
