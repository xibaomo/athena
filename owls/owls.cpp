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
/*
 * double binom_pdf(unsigned int k, double p, unsigned int n)
 * {
 *     return gsl_ran_binomial_pdf(k, p, n);
 * }
 */

static PyObject* binom_pdf(PyObject* self, PyObject* args)
{
    unsigned int k;
    unsigned int n;
    double p;

    PyArg_ParseTuple(args, "IId",&k, &n, &p);

    double pb = gsl_ran_binomial_pdf(k, p, n);

//    printf("%g\n",pb);

//    pb = pb==0?std::numeric_limits<double>::min():pb;

//    pb = gsl_sf_log(pb);

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
static PyMethodDef myMethods[] = {
    {"std",_std, METH_VARARGS, "compute std of array"},
    {"addone",_add_one, METH_VARARGS, "all elements of array adds one"},
    {"binom_pdf",binom_pdf, METH_VARARGS, "binomial pdf"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initowls() {
    import_array();
    Py_InitModule3("owls", myMethods, "owls doc");
}
