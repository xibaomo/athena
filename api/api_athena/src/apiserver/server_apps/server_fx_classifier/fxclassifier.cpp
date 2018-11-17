/*
 * =====================================================================================
 *
 *       Filename:  fxclassifier.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/16/2018 01:59:36 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "fxclassifier.h"
using namespace std;

void
ForexClassifier::prepare()
{
    loadPythonModules();
    loadFilterSet(m_fxSymbol,"buy");
    loadFilterSet(m_fxSymbol,"sell");
    configPredictor(m_buyPredictor);
    configPredictor(m_sellPredictor);
}

void
ForexClassifier::loadPythonModules()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/pyapi";
    m_pyInst.appendSysPath(modulePath);
    m_predictorModule = PyImport_ImportModule("forex_tick_predictor");
    if ( !m_predictorModule ) {
        Log(LOG_FATAL) << "Failed to import forex_tick_predictor";
    }

    CPyObject predictorClass = PyObject_GetAttrString(m_predictorModule,"ForexTickPredictor");
    if (!predictorClass)
        Log(LOG_FATAL) << "Failed to get predictor class";

    m_buyPredictor = PyObject_CallObject(predictorClass,NULL);
    if (!m_buyPredictor)
        Log(LOG_FATAL) << "Failed to create python buy predictor";
    m_sellPredictor = PyObject_CallObject(predictorClass,NULL);
    if (!m_sellPredictor)
        Log(LOG_FATAL) << "Failed to create python sell predictor";
}

void
ForexClassifier::loadFilterSet(const String& symbol, const String& pos_type)
{
    int numFilters = stoi(getYamlValue("PREDICTION/NUM_FILTERS"));
    for (int i = 0; i < numFilters; i++) {
        String mf = symbol + "_" + pos_type + "_" + to_string(i+1) + ".flt";
        CPyObject arg = Py_BuildValue("s",mf.c_str());
        if (pos_type == "buy") {
            PyObject_CallMethod(m_buyPredictor,"loadAModel","(O)",arg.getObject());
        } else if (pos_type == "sell") {
            PyObject_CallMethod(m_sellPredictor,"loadAModel","(O)",arg.getObject());
        } else {
            Log(LOG_FATAL) << "Unexpected position type: " + pos_type;
        }
    }
}

void
ForexClassifier::configPredictor(CPyObject& predictor)
{
    // set fast & slow periods
    int fastPeriod = stoi(getYamlValue("PREDICTION/FAST_PERIOD"));
    int slowPeriod = stoi(getYamlValue("PREDICTION/SLOW_PERIOD"));
    CPyObject arg1 = Py_BuildValue("i",fastPeriod);
    CPyObject arg2 = Py_BuildValue("i",slowPeriod);

    PyObject_CallMethod(predictor,"setPeriods","(OO)",arg1.getObject(),arg2.getObject());

    // set feature names
    String featureNames = getYamlValue("PREDICTION/FEATURE_LIST");
    CPyObject arg = Py_BuildValue("s",featureNames.c_str());
    PyObject_CallMethod(predictor,"setFeatureNames","(O)",arg.getObject());

    PyObject_CallMethod(predictor,"showFeatureCalculator",NULL);
}

Message
ForexClassifier::processMsg(Message& msg)
{
    Message msgnew;
    FXAction action = (FXAction)msg.getAction();
    switch(action) {
        case FXAction::HISTORY:
            msgnew = std::move(procMsg_HISTORY(msg));
            break;
        case FXAction::TICK:
            msgnew = std::move(procMsg_TICK(msg));
            break;
        case FXAction::CHECKIN:
            msgnew = std::move(procMsg_CHECKIN(msg));
        default:
            break;
    }

    return msgnew;
}

Message
ForexClassifier::procMsg_CHECKIN(Message& msg)
{
    Log(LOG_INFO) << "Client checks in";
    if (!compareStringNoCase(m_fxSymbol,msg.getComment()))
        Log(LOG_FATAL) << "Received symbol is inconsistent with model files";

    Message msgnew;
    return msgnew;
}

Message
ForexClassifier::procMsg_HISTORY(Message& msg)
{
    Log(LOG_INFO) << "Msg of history data received";

    int len = msg.getDataBytes()/sizeof(Real);
    CPyObject lst = PyList_New(len);
    Real* pm = (Real*)msg.getData();
    for (int i = 0; i < len; i++) {
        PyList_SetItem(lst,i,Py_BuildValue(REALFORMAT,pm[i]));
    }
    if (msg.getComment() == "buy") {
        PyObject_CallMethod(m_buyPredictor,(char*)"loadTicks","(O)",lst.getObject());
    } else if (msg.getComment() == "sell") {
        PyObject_CallMethod(m_sellPredictor,(char*)"loadTicks","(O)",lst.getObject());
    } else {
        Log(LOG_ERROR) << "Unexpected position type: " + msg.getComment();
    }

    Message msgnew;
    return msgnew;
}

Message
ForexClassifier::procMsg_TICK(Message& msg)
{
    Log(LOG_INFO) << "New tick arrives";
    Real *pm = (Real*)msg.getData();
    Real tick = pm[0];

    ActionType action;
    CPyObject pypred;
    CPyObject pytick = Py_BuildValue(REALFORMAT,tick);
    if (msg.getComment() == "buy") {
        pypred = PyObject_CallMethod(m_buyPredictor,"classifyATick","(O)",pytick.getObject());
        action = (ActionType)FXAction::PLACE_BUY;
    } else if (msg.getComment() == "sell") {
        pypred = PyObject_CallMethod(m_sellPredictor,"classifyATick","(O)",pytick.getObject());
        action = (ActionType)FXAction::PLACE_SELL;
    } else {
        Log(LOG_ERROR) << "Unexpected position type: " + msg.getComment();
        pypred = Py_BuildValue("i",1);
    }

    PyObject* objrepr = PyObject_Repr(pypred.getObject());
    const char* cp = PyString_AsString(objrepr);
    int pred = stoi(String(cp));
    Py_XDECREF(objrepr);
    if (pred == 1)
        action = (ActionType)FXAction::NOACTION;

    Message msgnew;
    msgnew.setAction(action);

    return msgnew;
}
