/*
 * =====================================================================================
 *
 *       Filename:  roblinreg.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/10/2019 04:47:00 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _LINREG_ROBUST_LINREG_H_
#define  _LINREG_ROBUST_LINREG_H_

#include <gsl/gsl_multifit.h>
#include <vector>
#include "basics/types.h"
struct RobLRParam {
    real64 c0;
    real64 c1;
    real64 sigma;
    real64 r2;
    real64 rms;
    real64 w_ave;
    real64 w_now;
};

RobLRParam robLinreg(std::vector<real32>& vx, std::vector<real32>& vy, size_t start=0);
RobLRParam robLinreg(real64* x, real64* y, size_t slen);

#endif   /* ----- #ifndef _LINREG_ROBUST_LINREG_H_  ----- */
