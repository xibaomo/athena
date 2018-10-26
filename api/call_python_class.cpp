/*
 * =====================================================================================
 *
 *       Filename:  call_python_class.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/26/2018 12:40:07 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include <iostream>
#include <string>
#include <python2.7/Python.h>
#include "pyhelper.hpp"
using namespace std;

int main() {
    CPyInstance hInstance;

    CPyObject pModule = PyImport_ImportModule("testpy");
    if ( !pModule ) {
        cout << "CAnnot open python file: testpy.py" << endl;
        return -1;
    }

    CPyObject pDict = PyModule_GetDict(pModule);
    if ( !pDict ) {
        cout << "Cannot find dict in file" << endl;
        return -1;
    }
    // call function
    CPyObject pFunHi = PyDict_GetItemString(pDict, "sayhi");
    PyObject_CallFunction(pFunHi, "d",10.);

    // call class
    CPyObject pClassSecond =
        PyDict_GetItemString(pDict, "Second");
    if ( !pClassSecond ) {
        cout << "Cannot find class" << endl;
        return -1;
    }
    CPyObject pClassPerson = PyDict_GetItemString(pDict, "Person");
    if ( !pClassPerson ) {
        cout << "Cannot find class" << endl;
        return -1;
    }

    // instance of class
    CPyObject pInsSecond = PyInstance_New(pClassSecond, NULL, NULL);
    CPyObject pInsPerson = PyInstance_New(pClassPerson, NULL, NULL);

    PyObject_CallMethod(pInsSecond, "invoke", "O",pInsPerson.getObject());
    return 0;
}
