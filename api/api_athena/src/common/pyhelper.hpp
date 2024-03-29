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
 *
 * =====================================================================================
 */

#ifndef  pyhelper_INC
#define  pyhelper_INC
#include <Python.h>
#include <iostream>
#include <sstream>
#include <string>
#include <set>
#include <numpy/arrayobject.h>

/**
 * Use a singleton to initialize and finalize python environment
 */

class PyEnviron {
private:
    std::set<std::string> m_addedModulePaths;
    PyEnviron() {
        Py_Initialize();
//        if (_import_array() < 0)
//            std::cerr << "Failed to import pyArray" << std::endl;
        PyRun_SimpleString("import sys");
    }
public:
    ~PyEnviron() {
        //Py_Finalize();
    }

    static PyEnviron& getInstance() {
        static PyEnviron _ins;
        return _ins;
    }

    void appendSysPath(const std::string& modulePath) {
        if(m_addedModulePaths.find(modulePath)!=m_addedModulePaths.end()) return;

        std::string cmd = "sys.path.append(\'"+modulePath + "\')";
        PyRun_SimpleString(cmd.c_str());
        PyRun_SimpleString("print(sys.path)");

        m_addedModulePaths.insert(modulePath);
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
