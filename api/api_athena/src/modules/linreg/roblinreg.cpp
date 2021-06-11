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

    int k=0;
    for(size_t i=start; i < vx.size(); i++) {
        gsl_vector_set(y,k,vy[i]);
        gsl_matrix_set(X,k,0,1.);
        gsl_matrix_set(X,k,1,vx[i]);
        k++;
    }

    gsl_multifit_robust_workspace *work =
        gsl_multifit_robust_alloc(gsl_multifit_robust_bisquare,X->size1,X->size2);

    // set max iter
    gsl_multifit_robust_maxiter(1000,work);

    int s= gsl_multifit_robust(X,y,c,cov,work);

    (void)s;

    gsl_multifit_robust_stats stat =
        gsl_multifit_robust_statistics(work);

    params.sigma = stat.sigma;
    params.rms   = stat.rmse;
    params.r2    = stat.Rsq;
    params.c0    = gsl_vector_get(c,0);
    params.c1    = gsl_vector_get(c,1);
    params.w_ave = gsl_stats_mean(stat.weights->data,1,len);
    params.w_now = stat.weights->data[len-1];

    gsl_vector_free(y);
    gsl_matrix_free(X);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
    gsl_multifit_robust_free(work);
    return params;
}

RobLRParam robLinreg(real64* sx, real64* sy, size_t len) {
    const size_t dim = 2;

    RobLRParam params;

    gsl_vector* c = gsl_vector_alloc(dim);
    gsl_matrix* cov = gsl_matrix_alloc(dim,dim);

    gsl_vector* y = gsl_vector_alloc(len);
    gsl_matrix* X = gsl_matrix_alloc(len,dim);

    int k=0;
    for(size_t i=0; i < len; i++) {
        gsl_vector_set(y,k,sy[i]);
        gsl_matrix_set(X,k,0,1.);
        gsl_matrix_set(X,k,1,sx[i]);
        k++;
    }

    gsl_multifit_robust_workspace *work =
        gsl_multifit_robust_alloc(gsl_multifit_robust_bisquare,X->size1,X->size2);

    // set max iter
    gsl_multifit_robust_maxiter(1000,work);

    int s= gsl_multifit_robust(X,y,c,cov,work);

    (void)s;

    gsl_multifit_robust_stats stat =
        gsl_multifit_robust_statistics(work);

    params.sigma = stat.sigma;
    params.rms   = stat.rmse;
    params.r2    = stat.Rsq;
    params.c0    = gsl_vector_get(c,0);
    params.c1    = gsl_vector_get(c,1);
    params.w_ave = gsl_stats_mean(stat.weights->data,1,len);
    params.w_now = stat.weights->data[len-1];

    gsl_vector_free(y);
    gsl_matrix_free(X);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
    gsl_multifit_robust_free(work);
    return params;
}

