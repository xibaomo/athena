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
struct LRParam {
    real64 c0, c1;
    real64 cov00, cov01, cov11;
    real64 chisq;
};

LRParam linreg(real64* x, real64* y, size_t n);

void linreg_est(LRParam& pm, real64 x, real64* yp, real64* sigma);
#endif   /* ----- #ifndef _LINREG_LINREG_H_  ----- */
