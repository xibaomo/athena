//*************************************************************************************//
//                         Copyright TTI-TSMC, 2015-2018                               //
//*************************************************************************************//
/*
 * =====================================================================================
 *
 *       Filename:  pyrunner.h
 *
 *    Description:  This class is used to launch python environment and run given scripts
 *
 *        Version:  1.0
 *        Created:  06/15/2019 04:54:20 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _COMMON_PYRUNNER_PYRUNNER_H_
#define  _COMMON_PYRUNNER_PYRUNNER_H_

#include "pyhelper.hpp"
#include "basics/types.h"
#include "basics/log.h"
class PyRunner {
protected:
    CPyInstance m_pyInst;
    PyRunner();
public:
    virtual ~PyRunner(){;}

    static PyRunner& getInstance() {
        static PyRunner _ins;
        return _ins;
    }

    CPyObject runAthenaPyFunc(const String& modName, const String& funcName, CPyObject& args);
};

#endif   /* ----- #ifndef _COMMON_PYRUNNER_PYRUNNER_H_  ----- */
