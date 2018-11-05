/*
 * =====================================================================================
 *
 *       Filename:  server_predictor.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/04/2018 17:28:23
 *
 *         Author:  fxua (), fxua@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "server_predictor.h"
#include "predictor/prdtypes.h"
#include <iostream>
using namespace std;

void
ServerPredictor::prepare()
{
    loadPythonModule();
}

void
ServerPredictor::loadPythonModule()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/apps/generic_app";
    m_pyInst.appendSysPath(modulePath);

    modulePath = athenaHome + "/modules/mlengines/regressor";
    m_pyInst.appendSysPath(modulePath);

    m_engCreatorMod = PyImport_ImportModule("engine_creator");
    m_mlEngMod = PyImport_ImportModule("regressor");

    modulePath = athenaHome + "/modules/mlengine_cores";
    m_pyInst.appendSysPath(modulePath);
    m_engineCoreMod = PyImport_ImportModule("mlengine_core");

    if ( !m_engineCoreMod ) {
        Log(LOG_FATAL) = "Failed to import module of engine core";
    }
}

void
ServerPredictor::loadEngine(EngineType et, EngineCoreType ect, const String& mf)
{
    // create engine core
    CPyObject coreClass = PyObject_GetAttrString(m_engineCoreMod, "MLEngineCore");
    if ( !coreClass ) {
        Log(LOG_ERROR) << "Failed to get engine core class";
    }

    // m_engineCore = PyInstance_New(coreClass.getObject(), NULL, NULL);
    m_engineCore = PyObject_CallObject(coreClass, NULL);
    if ( !m_engineCore ) {
        Log(LOG_ERROR) << "Failed to create engine core";
    }

    CPyObject arg1 = Py_BuildValue("i",ect);
    CPyObject arg2 = Py_BuildValue("s",mf.c_str());
    PyObject_CallMethod(m_engineCore, "loadModel","(OO)",arg1.getObject(),
            arg2.getObject());

    Log(LOG_INFO) << "-------------- ML engine core created -------------";
    PyObject_CallMethod(m_engineCore, "showEstimator",NULL);

    // create engine
    CPyObject engClass = PyObject_GetAttrString(m_mlEngMod, "Regressor");
    if ( !engClass ) {
        Log(LOG_ERROR) << "Failed to get engine class";
    }

    m_engine = PyObject_CallObject(engClass, NULL);
    if ( !m_engine ) {
        Log(LOG_ERROR) << "Failed to create ML engine";
    }

    PyObject_CallMethod(m_engine, "loadEngineCore","(O)",m_engineCore.getObject());
}

void
ServerPredictor::predict(Real* featureMatrix, const Uint rows, const Uint cols)
{
    // create python list to pass array
    CPyObject lst = PyList_New(rows*cols);
    for ( Uint i = 0; i < rows*cols; i++ ) {
        PyList_SetItem(lst, i, Py_BuildValue(REALFORMAT, featureMatrix[i]));
    }

//    CPyObject args = PyTuple_New(3);
//    PyTuple_SetItem(args, 0, lst.getObject());
//    CPyObject val = PyInt_FromSize_t(rows);
//    PyTuple_SetItem(args, 1, val.getObject());
//    val = PyInt_FromSize_t(cols);
//    PyTuple_SetItem(args, 2, val.getObject());
    CPyObject arg1 = Py_BuildValue("i",rows);

    char* funcName = (char*)"predict_array";
    PyObject_CallMethod(m_engine, funcName, "(OOi)",lst.getObject(),
            arg1.getObject(), cols);

    CPyObject preds = PyObject_CallMethod(m_engine, (char*)"getPredictedTargets",NULL);

    if ( !preds ) {
        Log(LOG_FATAL) << "Failed to get result from python";
    }

    PyArrayObject* np_res = reinterpret_cast<PyArrayObject*>(preds.getObject());
    Uint len = np_res->dimensions[0];
    if ( len != rows ) {
        Log(LOG_FATAL) << "Returned prediction size inconsistent with sent samples";
        return;
    }

    m_result_array = reinterpret_cast<double*>(PyArray_DATA(np_res));

    sendBackResult((MsgAction)PrdAction::RESULT, m_result_array, len);
    return;
}

void
ServerPredictor::processMsg(Message& msg)
{
    PrdAction action = (PrdAction)msg.getAction();
    switch(action) {
        case PrdAction::SETUP:
            procMsg_SETUP(msg);
            break;
        case PrdAction::PREDICT:
            procMsg_PREDICT(msg);
            break;
        default:
            break;
    }
}

void
ServerPredictor::procMsg_SETUP(Message& msg)
{
    Log(LOG_INFO) << "Msg of model file received";
    int *pm = (int*) msg.getData();
    EngineType engtype = (EngineType)pm[0];
    EngineCoreType coretype = (EngineCoreType)pm[1];
    String mf = msg.getComment();

    loadEngine(engtype, coretype, mf);
}

void
ServerPredictor::procMsg_PREDICT(Message& msg)
{
    Log(LOG_DEBUG) << "Features received";
    Real* pf = (Real*)msg.getData();
    Uint* pc = (Uint*)msg.getChar();
    Uint n1 = pc[0];
    Uint n2 = pc[1];

    predict(pf, n1, n2);
}