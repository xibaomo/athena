/*
 * =====================================================================================
 *
 *       Filename:  pyhelper.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/25/2018 10:28:31 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef  pyhelper_INC
#define  pyhelper_INC
#include <python2.7/Python.h>

class CPyInstance {
public:
    CPyInstance() {
        Py_Initialize();
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('./')");
    }
    ~CPyInstance() {
        Py_Finalize();
    }
};

class CPyObject {
private:
    PyObject *p;
public:
    CPyObject() : p(nullptr) {}
    CPyObject(PyObject* _p) : p(_p) {}
    ~CPyObject() {
        release();
    }

    PyObject* getObject() {
        return p;
    }

    PyObject* setObject(PyObject* _p) { return p = _p;}

    PyObject* addRef() {
        if (p) {
            Py_INCREF(p);
        }
        return p;
    }

    void release() {
        if (p) {
            Py_DECREF(p);
        }
        p = nullptr;

    }

    PyObject* operator->() {
        return p;
    }

    bool is() {
        return p? true : false;
    }

    PyObject* operator= (PyObject* pp) {
        p = pp;
        return p;
    }


    // below are converter operators
    operator PyObject*() {
        return p;
    }

    operator bool() {
        return p? true : false;
    }
};
#endif   /* ----- #ifndef pyhelper_INC  ----- */
