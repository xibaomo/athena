/*
 * =====================================================================================
 *
 *       Filename:  pair_asset_meanrevert.cpp
 *
 *    Description:  the paired symbols satisfy y = a*x + b + epsilon(x), equivalently, we are trading
 *                  a custom symbol (y - a*x), which is stationary.
 *                  In most cases, when lot size of y and/or x is quite small, the ratio of y and x hardly
 *                  ensures (y-a*x) is stationary.
 *                  This concrete oracle, will carefully select lot sizes of both x and y, so as to ensure
 *                  (lot_y*y +/- lot_x*x) is stationary, and the rest is still classical mean-reverting algorithm.
 *
 *
 *        Version:  1.0
 *        Created:  07/15/2021 03:56:03 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "pair_asset_meanrevert.h"
#include "basics/utils.h"
#include "linreg/linreg.h"
#include <gsl/gsl_multimin.h>
using namespace std;
using namespace athena;

static real64 ratio_cost(const gsl_vector* v, void* params) {
    real64 x = gsl_vector_get(v,0);
    real64 y = gsl_vector_get(v,1);
    real64* p = (real64*)params;

    real64 c = (y/x - p[0]);
    return c*c;
}

static
void ratio_cost_df(const gsl_vector* v, void* params, gsl_vector* df) {
    real64 x = gsl_vector_get(v,0);
    real64 y = gsl_vector_get(v,1);
    real64* p = (real64*)params;

    real64 gx = 2.f*(y/x-p[0])*(-y/x/x);
    real64 gy = 2.f*(y/x-p[0])*1.f/x;
    gsl_vector_set(df,0,gx);
    gsl_vector_set(df,1,gy);
}

static
void ratio_cost_fdf(const gsl_vector* v, void* params, real64* f, gsl_vector* df) {
    *f = ratio_cost(v,params);
    ratio_cost_df(v,params,df);
}
static
void minimize_ratio_cost(real64 x0, real64 y0, real64 target) {
    gsl_vector* x = gsl_vector_alloc(2);
    gsl_multimin_function_fdf func;
    func.n = 2;
    func.f = ratio_cost;
    func.df = ratio_cost_df;
    func.fdf = ratio_cost_fdf;
    func.params = &target;

    gsl_vector_set(x,0,x0);
    gsl_vector_set(x,1,y0);

    int status,iter(0);
    const gsl_multimin_fdfminimizer_type* T = gsl_multimin_fdfminimizer_conjugate_fr;
    gsl_multimin_fdfminimizer* s = gsl_multimin_fdfminimizer_alloc(T,2);
    gsl_multimin_fdfminimizer_set(s,&func,x,0.01,1e-4);

    do {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(s);
        if(status) break;

        status = gsl_multimin_test_gradient(s->gradient,1e-3);
        if(status==GSL_SUCCESS) {
            ostringstream oss;
            oss << "Minimum found: " << s->f << " at (" << gsl_vector_get(s->x,0) << "," << gsl_vector_get(s->x,1)<<")";
            Log(LOG_INFO) << oss.str();
        }
    }while(status==GSL_CONTINUE && iter <500);

    real64 xp = gsl_vector_get(s->x,0);
    real64 yp = gsl_vector_get(s->x,1);
    xp = floor(xp*100.f)/100.f;
    yp = floor(yp*100.f)/100.f;

    ostringstream oss;
    oss << "lot_x: " << xp << ", lot_y: " << yp << ", lot_y/log_x = " << yp/xp;
    Log(LOG_INFO) << oss.str();
    gsl_vector_free(x);
    gsl_multimin_fdfminimizer_free(s);
}
void
PairAssetMeanRevert::findBestLots() {
    auto& asset_x = m_trader->getAssetX();
    auto& asset_y = m_trader->getAssetY();

    LRParam param = linreg(&asset_x[0],&asset_y[0],asset_x.size());

    minimize_ratio_cost(0.5,0.5,abs(param.c1));
}
void
PairAssetMeanRevert::init() {
}

FXAct
PairAssetMeanRevert::getDecision() {
    return FXAct::NOACTION;
}

bool
PairAssetMeanRevert::isContinue() {
    return true;
}
