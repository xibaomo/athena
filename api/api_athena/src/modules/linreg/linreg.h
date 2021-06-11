/*
 * =====================================================================================
 *
 *       Filename:  linreg.h
 *
 *    Description:  Linear regression
 *
 *        Version:  1.0
 *        Created:  05/27/2019 04:37:30 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _LINREG_LINREG_H_
#define  _LINREG_LINREG_H_

#include "basics/types.h"
#include <vector>
#include <gsl/gsl_statistics.h>
struct LRParam {
    real64 c0, c1;
    real64 cov00, cov01, cov11;
    real64 chisq;
    real64 r2;
};

LRParam linreg(real64* x, real64* y, size_t n);
LRParam ordLinreg(std::vector<real32>& x, std::vector<real32>& y);

void linreg_est(LRParam& pm, real64 x, real64* yp, real64* sigma);

template <typename T>
real64 compR2(T& pm, real64* x, real64* y, int len)
{
    real64 ss_tot=0.;
    real64 ss_res=0.;
    real64 mean_y = gsl_stats_mean(y,1,len);
    for (int i=0; i < len; i++) {
        ss_tot += (y[i]-mean_y)*(y[i]-mean_y);
        real64 err = pm.c0 + pm.c1*x[i] - y[i];
        ss_res += err*err;
    }
    real64 r2 = 1-ss_res/ss_tot;

    return r2;
}
#endif   /* ----- #ifndef _LINREG_LINREG_H_  ----- */
