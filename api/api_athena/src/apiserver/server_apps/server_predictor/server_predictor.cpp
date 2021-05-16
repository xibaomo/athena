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
#include "fx_action/fx_action.h"
#include "basics/utils.h"
#include <iostream>
using namespace std;
using namespace athena;

void
ServerPredictor::prepare()
{
    loadPythonModule();
    loadAllFilters();
}

void
ServerPredictor::loadAllFilters()
{
    m_fxSymbol = getYamlValue("PREDICTION/SYMBOL");

    loadFilterSet(m_fxSymbol,"buy");
    loadFilterSet(m_fxSymbol,"sell");

}

void
ServerPredictor::loadFilterSet(const String& symbol, const String& pos_type)
{
    int numFilters = stoi(getYamlValue("PREDICTION/NUM_FILTERS"));
    for (int i = 0; i< numFilters; i++) {
        String mf = symbol + "_" + pos_type + "_" + to_string(i+1) + ".flt";
        loadEngine((EngineType)0, (EngineCoreType)2,mf);
        if(pos_type == "buy")
            m_buyEngines.push_back(m_engine);
        if (pos_type == "sell")
            m_sellEngines.push_back(m_engine);
    }
}

void
ServerPredictor::loadPythonModule()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/apps/generic_app";
    PyEnviron::getInstance().appendSysPath(modulePath);

    modulePath = athenaHome + "/modules/mlengines/regressor";
    PyEnviron::getInstance().appendSysPath(modulePath);

    m_engCreatorMod = PyImport_ImportModule("engine_creator");
    m_mlEngMod = PyImport_ImportModule("regressor");

    modulePath = athenaHome + "/modules/mlengine_cores";
    PyEnviron::getInstance().appendSysPath(modulePath);
    m_engineCoreMod = PyImport_ImportModule("mlengine_core");

    if ( !m_engineCoreMod ) {
        Log(LOG_FATAL) << "Failed to import module of engine core";
    }
}

void
ServerPredictor::loadEngine(EngineType et, EngineCoreType ect, const String& mf)
{
    // create engine core
    CPyObject coreClass = PyObject_GetAttrString(m_engineCoreMod, "MLEngineCore");
    if ( !coreClass ) {
        Log(LOG_ERROR) << "Failed to get engine core class";
        return;
    }

    // m_engineCore = PyInstance_New(coreClass.getObject(), NULL, NULL);
    m_engineCore = PyObject_CallObject(coreClass, NULL);
    if ( !m_engineCore ) {
        Log(LOG_ERROR) << "Failed to create engine core";
        return;
    }

    CPyObject arg1 = Py_BuildValue("i",ect);
    CPyObject arg2 = Py_BuildValue("s",mf.c_str());
    PyObject_CallMethod(m_engineCore, "loadModel","(OO)",arg1.getObject(),
            arg2.getObject());

    Log(LOG_INFO) << "ML engine core created from " + mf;
    PyObject_CallMethod(m_engineCore, "showEstimator",NULL);

    // create engine
    CPyObject engClass = PyObject_GetAttrString(m_mlEngMod, "Regressor");
    if ( !engClass ) {
        Log(LOG_ERROR) << "Failed to get engine class";
        return;
    }

    m_engine = PyObject_CallObject(engClass, NULL);
    if ( !m_engine ) {
        Log(LOG_ERROR) << "Failed to create ML engine";
        return;
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

//    sendBackResult((MsgAct)PrdAction::RESULT, m_result_array, len);
    return;
}

Message
ServerPredictor::processMsg(Message& msg)
{
    Message msgnew;
    FXAct action = (FXAct)msg.getAction();
    switch(action) {
        case FXAct::HISTORY:
            msgnew = std::move(procMsg_HISTORY(msg));
            break;
        case FXAct::TICK:
            msgnew = std::move(procMsg_TICK(msg));
            break;

        default:
            break;
    }

    return msgnew;
}

Message
ServerPredictor::procMsg_CHECKIN(Message& msg)
{
    Log(LOG_INFO) << "Client checks in";
    if (!compareStringNoCase(m_fxSymbol,msg.getComment()))
        Log(LOG_FATAL) << "Received symbol is inconsistent with model files";

    Message msgnew;
    return msgnew;
}

Message
ServerPredictor::procMsg_HISTORY(Message& msg)
{
    Log(LOG_INFO) << "Msg of history data received";
    int len = msg.getDataBytes() / sizeof(Real);
    if (msg.getComment() == "buy") {
        m_buyTicks.resize(len);
        memcpy(&m_buyTicks[0],msg.getData(),msg.getDataBytes());
    } else if (msg.getComment() == "sell") {
        m_sellTicks.resize(len);
        memcpy(&m_sellTicks[0],msg.getData(),msg.getDataBytes());
    } else {
        Log(LOG_FATAL) << "Unknown position type: " + msg.getComment();
    }

    Message msgnew;
    return msgnew;
}

Message
ServerPredictor::procMsg_TICK(Message& msg)
{
    Log(LOG_INFO) << "New tick arrives";
    Message msgnew;


    return msgnew;
}
