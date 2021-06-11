/*
 * =====================================================================================
 *
 *       Filename:  linreg.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  05/27/2019 05:42:21 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "linreg.h"
#include "basics/utils.h"
#include <gsl/gsl_fit.h>
using namespace std;

LRParam linreg(real64* xs, real64* ys, size_t n)
{
    LRParam pm;
    gsl_fit_linear(xs, 1, ys, 1, n, &pm.c0, &pm.c1, &pm.cov00, &pm.cov01, &pm.cov11, &pm.chisq);

    pm.r2 = compR2(pm,xs,ys,n);
    return pm;
}

LRParam ordLinreg(std::vector<real32>& vx, std::vector<real32>& vy)
{
    int len = vx.size();
    real64* x = new real64[len];
    real64* y = new real64[len];
    for (int i =0; i < len; i++){
        x[i] = vx[i];
        y[i] = vy[i];
    }

    LRParam pm = linreg(x,y,len);

    delete[] x;
    delete[] y;

    return pm;

}

void linreg_est(LRParam& pm, real64 xp, real64* yp, real64* yp_err)
{
    gsl_fit_linear_est(xp,pm.c0,pm.c1,pm.cov00,pm.cov01,pm.cov11,yp,yp_err);
}


