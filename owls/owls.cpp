/*
 * =====================================================================================
 *
 *       Filename:  owls.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  02/17/2019 04:38:41 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include <python2.7/Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <numeric>
#include <iterator>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_log.h>
#include <limits>
#include <cstdio>
#include <iostream>
using namespace std;
typedef unsigned int Uint;
void pyArray2vector(PyArrayObject* pyarr, vector<double>& v)
{
    v.clear();
    double* p = (double*)pyarr->data;

    v.assign(p, p+pyarr->dimensions[0]);
}

static PyObject* binom_entropy(PyObject* self, PyObject* args)
{
    unsigned int k;
    unsigned int n;
    double p;

    PyArg_ParseTuple(args, "IId",&k, &n, &p);

    double pb = gsl_ran_binomial_pdf(k, p, n);

    if ( pb == 0) return PyFloat_FromDouble(pb);

    pb = -pb*gsl_sf_log(pb);

    return PyFloat_FromDouble(pb);
}

static PyObject* binom_pdf(PyObject* self, PyObject* args)
{
    unsigned int k;
    unsigned int n;
    double p;

    PyArg_ParseTuple(args, "IId",&k, &n, &p);

    double pb = gsl_ran_binomial_pdf(k, p, n);

    return PyFloat_FromDouble(pb);
}

static PyObject* binom_logpdf(PyObject* self, PyObject* args)
{
    unsigned int k;
    unsigned int n;
    double p;

    PyArg_ParseTuple(args, "IId",&k, &n, &p);

    double pb = gsl_ran_binomial_pdf(k, p, n);

    pb = pb==0?std::numeric_limits<double>::min(): pb;

    pb = gsl_sf_log(pb);

    return PyFloat_FromDouble(pb);
}

void binom(std::vector<double>& invec, std::vector<double>& outvec)
{
}
double std_std(std::vector<double>& v)
{
    double sum = std::accumulate(v.begin(), v.end(), 0.);
    double mean = sum/v.size();
    double squaredsum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);

    return sqrt(squaredsum/v.size() - mean*mean);
}

static PyObject* _std(PyObject* self, PyObject* args)
{
    PyObject* input;
    PyArg_ParseTuple(args, "O",&input);
    PyArrayObject* array = (PyArrayObject*)input;
    if ( !array ) {
        printf("failed to convert pyobject to pyarray\n");
        return NULL;
    }

    std::vector<double> v;
    double* p = (double*)array->data;
    v.assign(p, p+array->dimensions[0]);

//    return PyFloat_FromDouble(std_std(v));
    return PyFloat_FromDouble(NAN);
}

static PyObject* _add_one(PyObject* self, PyObject* args)
{
    PyObject* input;
    PyArg_ParseTuple(args, "O",&input);
    PyArrayObject* array = (PyArrayObject*)input;
    if ( !array ) {
        printf("Failed to convert pyobject to pyarrayobject\n");
        return NULL;
    }

//    printf("array parsed: %d\n",array->dimensions[0]);

    PyArrayObject* out = NULL;
    int dim[1];
    dim[0] = array->dimensions[0];
//    dim[1] = 1;
//    printf("old array: %d\n",dim[0]);

    out = (PyArrayObject*)PyArray_FromDims(1, dim, NPY_DOUBLE);
//    out = (PyArrayObject*)PyArray_SimpleNew(1, dim, NPY_FLOAT);

//    printf("out array created: %d\n",out->dimensions[0]);

    double* outdata = (double*)out->data;
    double* indata  = (double*)array->data;
    for ( int i = 0; i < dim[0]; i++ ) {
        outdata[i] = indata[i] +1;
    }

    return PyArray_Return(out);
}

static PyObject* _binom(PyObject* self, PyObject* args)
{
    PyObject* input;
    if ( !PyArg_ParseTuple(args, "O",&input) )
        return NULL;

    PyArrayObject* arr = NULL;
    PyArrayObject* out = NULL;
    arr = (PyArrayObject*)input;
    int dims[1];
    dims[0] = arr->dimensions[0];
    out = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);
    std::vector<double> invec;
    std::vector<double> outvec;
    double *pin = (double*)arr->data;
    double *pou = (double*)out->data;
    invec.assign(pin, pin+dims[0]);
    outvec.assign(pou, pou+dims[0]);
    binom(invec, outvec);

    return PyArray_Return(out);
}

static PyObject* _getNan(PyObject* self, PyObject* args)
{
    return PyFloat_FromDouble(NAN);
}

/*-----------------------------------------------------------------------------
 *  Compute binomial features.
 *  Current label is included.
 *  Return: 1. ratio of 1 in lookback bars.
 *          2. probability of k occurrances in lookback bars, global probability
 *             is computed over prob_period
 *-----------------------------------------------------------------------------*/
void compBinomFeature(PyArrayObject* pylabels, Uint lookback, Uint prob_period,
        PyArrayObject* pyratio, PyArrayObject* pyprobs)
{
    double* labels = (double*)pylabels->data;
    double* ratio  = (double*)pyratio->data;
    double* probs  = (double*)pyprobs->data;

#pragma omp parallel for num_threads(20)
    for ( int i =0; i < pylabels->dimensions[0]; i++ ) {
        if ( i < prob_period ) {
            ratio[i] = NAN;
            probs[i] = NAN;
            continue;
        }
        // count 1 and -1 in prob_period
        int  k1 = 0;
        int  kn1 = 0;
        for ( int j = i - prob_period; j <= i; j++ ) {
            if ( labels[j] == 1) k1++;
            if ( labels[j] == -1) kn1++;
        }
        double p = k1*1./(prob_period+1.-kn1);

        k1 = 0; kn1 = 0;
        for ( int j = i - lookback; j<=i; j++ ) {
            if ( labels[j] == 1) k1++;
            if ( labels[j] == -1) kn1++;
        }
        double pb = gsl_ran_binomial_pdf(k1, p, lookback-kn1);
        pb = pb==0?std::numeric_limits<double>::min(): pb;

        probs[i] = -gsl_sf_log(pb);
        ratio[i] = k1*1./(lookback+1.-kn1);
    }
}
void compLastBinom(PyArrayObject* pylabels, Uint lookback, Uint prob_period,
        double& ratio, double& prob)
{
    if ( pylabels->dimensions[0] + 1 < prob_period ) {
        ratio = NAN;
        prob = NAN;
        return;
    }
    // count 1 and -1 in prob_period
    int i = pylabels->dimensions[0]-1;
    double* labels = (double*)pylabels->data;
    int  k1 = 0;
    int  kn1 = 0;
    for ( int j = i - prob_period; j <= i; j++ ) {
        if ( labels[j] == 1) k1++;
        if ( labels[j] == -1) kn1++;
    }
    double p = k1*1./(prob_period+1.-kn1);

    k1 = 0; kn1 = 0;
    for ( int j = i - lookback; j<=i; j++ ) {
        if ( labels[j] == 1) k1++;
        if ( labels[j] == -1) kn1++;
    }
    double pb = gsl_ran_binomial_pdf(k1, p, lookback-kn1);
    pb = pb==0?std::numeric_limits<double>::min(): pb;

    prob = -gsl_sf_log(pb);
    ratio = k1*1./(lookback+1.-kn1);
}

/*-----------------------------------------------------------------------------
 *  Compute binom features of the entire label array
 *-----------------------------------------------------------------------------*/
static PyObject* _compBinom(PyObject* self, PyObject* args)
{
    PyObject* in_arr;
    PyArrayObject* arr = NULL;
    unsigned int lookback;
    unsigned int prob_period;
    if ( !PyArg_ParseTuple(args, "OII",&arr, &lookback, &prob_period) ) {
        return NULL;
    }

    int dims[1];
    dims[0] = arr->dimensions[0];
    PyArrayObject* o1 = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);
    PyArrayObject* o2 = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);

    compBinomFeature(arr, lookback, prob_period, o1, o2);
    return Py_BuildValue("OO",o1, o2);
}

/*-----------------------------------------------------------------------------
 *  Compute binom feature of the last label
 *-----------------------------------------------------------------------------*/
static PyObject* _compLastBinom(PyObject* self, PyObject* args)
{
    Uint lookback;
    Uint prob_period;
    PyArrayObject* arr;

    if ( !PyArg_ParseTuple(args, "OII",&arr, &lookback, &prob_period) ) {
        return NULL;
    }
    double ratio;
    double prob;
    compLastBinom(arr, lookback, prob_period, ratio, prob);

    return Py_BuildValue("dd",ratio, prob);
}

/*-----------------------------------------------------------------------------
 *  Compute RSI feature of the entire history
 *  Returns: 1. the accumulated net increase points
 *           2. the ratio of increase points over increase + decrease points
 *-----------------------------------------------------------------------------*/
void __compRSI(PyArrayObject* open, PyArrayObject* close, Uint lookback, double digits,
        PyArrayObject* inc_pts, PyArrayObject* dec_pts, PyArrayObject* r_inc_pts)
{
    int hist_len = open->dimensions[0];
    double* pinc = (double*)inc_pts->data;
    double* pdec = (double*)dec_pts->data;
    double* pr   = (double*)r_inc_pts->data;

    double* po = (double*)open->data;
    double* pc = (double*)close->data;

#pragma omp parallel for num_threads(20)
    for ( int i =0; i < hist_len; i++ ) {
        if ( i < lookback - 1 ) {
            pinc[i] = NAN;
            pr[i] = NAN;
            continue;
        }
        vector<int> inc;
        vector<int> dec;
        for ( int j = i - lookback + 1; j <= i; j++ ) {
            double ch = pc[j] - po[j];
            int pts = ch / digits;
            if ( pts >=0 ) {
                inc.push_back(pts);
            } else {
                dec.push_back(-pts);
            }
        }
        double sum_up = std::accumulate(inc.begin(), inc.end(), 0.);
        double sum_down = std::accumulate(dec.begin(), dec.end(), 0.);
        pinc[i] = sum_up - sum_down;
//        pinc[i] = sum_up;
        pdec[i] = sum_down;
        pr[i] = sum_up/(sum_up+sum_down);
    }
}
static PyObject* _compRSI(PyObject* self, PyObject* args)
{
    Uint lookback;
    PyArrayObject* open;
    PyArrayObject* close;
    double digits;

    if ( !PyArg_ParseTuple(args, "OOId",&open, &close, &lookback, &digits) ) {
        return NULL;
    }

    int dims[1];
    dims[0] = open->dimensions[0];
    PyArrayObject* inc_pts = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);
    PyArrayObject* dec_pts = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);
    PyArrayObject* r_inc_pts = (PyArrayObject*)PyArray_FromDims(1, dims, NPY_DOUBLE);

    __compRSI(open, close, lookback, digits, inc_pts, dec_pts, r_inc_pts);

    return Py_BuildValue("OOO",inc_pts, dec_pts, r_inc_pts);
}

/*-----------------------------------------------------------------------------
 *  Find the consecutive increasing subarray with the largest climb-up
 *-----------------------------------------------------------------------------*/
template <typename Fn>
void __maxGapSubArray(const Fn& fn, PyArrayObject* pyarr,
        double& max_gap, int& startID, int& endID) //endID not included
{
    size_t i = 0;
    max_gap = 0.;
    size_t len = pyarr->dimensions[0];
    double* arr = (double*)pyarr->data;
    while ( i < len ) {
        size_t j = i+1;
        while ( j < len ) {
            if ( fn( j) ) {
                double gap = fabs(arr[j-1] - arr[i]);
                if ( gap > max_gap ) {
                    max_gap = gap;
                    startID = i;
                    endID = j;
                }
                i  = j;

                break;
            }
            j++;
        }
        if ( j == len ) {
            double gap = fabs(arr[j-1] - arr[i]);
            if ( gap > max_gap ) {
                max_gap = gap;
                startID = i;
                endID = j;
            }
            break;
        }
    }
}

void __maxRiseSubArray(PyArrayObject* pyarr,
        double& max_rise, int& sid, int& eid)
{
    __maxGapSubArray([&pyarr](size_t j){ double* arr = (double*)pyarr->data;return arr[j] < arr[j-1]; }, pyarr, max_rise, sid, eid);
}

void __maxDropSubArray(PyArrayObject* pyarr,
        double& max_drop, int& sid, int& eid)
{
    __maxGapSubArray([&pyarr](size_t j){ double* arr = (double*)pyarr->data;return arr[j] > arr[j-1]; }, pyarr, max_drop, sid, eid);
}

static PyObject* _maxRise(PyObject* self, PyObject* args)
{
    PyArrayObject* arr;
    if ( !PyArg_ParseTuple(args, "O",&arr) ) {
        return NULL;
    }
    int sid, eid;
    double mrise;
    __maxRiseSubArray(arr, mrise, sid, eid);

    return Py_BuildValue("dii",mrise, sid, eid);
}

//=============================================
// Method list
//=============================================
static PyMethodDef myMethods[] = {
    {"std",_std, METH_VARARGS, "compute std of array"},
    {"addone",_add_one, METH_VARARGS, "all elements of array adds one"},
    {"binom_entropy",binom_entropy, METH_VARARGS, "binomial entropy"},
    {"binom_pdf",binom_pdf, METH_VARARGS, "binomial pdf"},
    {"binom_logpdf",binom_logpdf, METH_VARARGS, "binomial log pdf"},
    {"getNan",_getNan, METH_NOARGS, "GET A NAN"},
    {"compBinom",_compBinom, METH_VARARGS, "WFWEOJO"},
    {"compLastBinom",_compLastBinom, METH_VARARGS, "only compute binom of last label"},
    {"compRSI",_compRSI, METH_VARARGS, "RSI feature"},
    {"maxRise",_maxRise, METH_VARARGS, "max-rise sub-array"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initowls() {
    import_array();
    Py_InitModule3("owls", myMethods, "owls doc");
}
