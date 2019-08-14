/*
 * =====================================================================================
 *
 *       Filename:  roblinreg.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/10/2019 04:50:20 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "roblinreg.h"
#include "basics/log.h"
#include <gsl/gsl_statistics.h>
using namespace std;

RobLRParam
robLinreg(vector<real32>& vx, vector<real32>& vy, size_t start)
{
    const size_t dim = 2;
    const size_t len = vx.size() - start;
    RobLRParam params;

    gsl_vector* c = gsl_vector_alloc(dim);
    gsl_matrix* cov = gsl_matrix_alloc(dim,dim);

    gsl_vector* y = gsl_vector_alloc(len);
    gsl_matrix* X = gsl_matrix_alloc(len,dim);

    for(size_t i=start; i < vx.size(); i++) {
        gsl_vector_set(y,i,vy[i]);
        gsl_matrix_set(X,i,0,1.);
        gsl_matrix_set(X,i,1,vx[i]);
    }

    gsl_multifit_robust_workspace *work =
        gsl_multifit_robust_alloc(gsl_multifit_robust_bisquare,X->size1,X->size2);
    int s= gsl_multifit_robust(X,y,c,cov,work);

    (void)s;

    gsl_multifit_robust_stats stat =
        gsl_multifit_robust_statistics(work);

    params.sigma = stat.sigma;
    params.rms   = stat.rmse;
    params.r2    = stat.Rsq;
    params.c0    = gsl_vector_get(c,0);
    params.c1    = gsl_vector_get(c,1);
    real64 ave_w = gsl_stats_mean(stat.weights->data,1,vx.size());
    params.w_ave = ave_w;
    params.w_now = stat.weights->data[vx.size()-1];

    gsl_vector_free(y);
    gsl_matrix_free(X);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
    gsl_multifit_robust_free(work);
    return params;
}
