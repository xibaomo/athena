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

#include "fx_tick_classifier.h"
using namespace std;
using namespace athena;
void
ForexTickClassifier::prepare()
{
    loadPythonModules();

    //String mf = getYamlValue("PREDICTION/BUY_MODEL");
    //loadFilter(m_buyPredictor, mf);

//    mf = getYamlValue("PREDICTION/SELL_MODEL");
//    loadFilter(m_sellPredictor, mf);

//    int fp = stoi(getYamlValue("PREDICTION/BUY_FAST_PERIOD"));
//    int sp = stoi(getYamlValue("PREDICTION/BUY_SLOW_PERIOD"));
//    configPredictor(m_buyPredictor, fp, sp);
//
//    fp = stoi(getYamlValue("PREDICTION/SELL_FAST_PERIOD"));
//    sp = stoi(getYamlValue("PREDICTION/SELL_SLOW_PERIOD"));
//    configPredictor(m_sellPredictor, fp, sp);
}

void
ForexTickClassifier::loadPythonModules()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/pyapi";
    PyEnviron::getInstance().appendSysPath(modulePath);
    m_predictorModule = PyImport_ImportModule("forex_tick_predictor");
    if ( !m_predictorModule ) {
        Log(LOG_FATAL) << "Failed to import forex_tick_predictor" <<std::endl;
    }

    CPyObject predictorClass = PyObject_GetAttrString(m_predictorModule, "ForexTickPredictor");
    if ( !predictorClass )
        Log(LOG_FATAL) << "Failed to get predictor class" <<std::endl;

    m_buyPredictor = PyObject_CallObject(predictorClass, NULL);
    if ( !m_buyPredictor )
        Log(LOG_FATAL) << "Failed to create python buy predictor" <<std::endl;
    m_sellPredictor = PyObject_CallObject(predictorClass, NULL);
    if ( !m_sellPredictor )
        Log(LOG_FATAL) << "Failed to create python sell predictor" <<std::endl;
}

void
ForexTickClassifier::loadFilter(CPyObject& predictor, const String& modelFile)
{
    CPyObject arg = Py_BuildValue("s",modelFile.c_str());
    PyObject_CallMethod(predictor, "loadAModel","(O)",arg.getObject());
}

void
ForexTickClassifier::configPredictor(CPyObject& predictor, int fastPeriod, int slowPeriod)
{
    // set fast & slow periods
    CPyObject arg1 = Py_BuildValue("i",fastPeriod);
    CPyObject arg2 = Py_BuildValue("i",slowPeriod);

    PyObject_CallMethod(predictor, "setPeriods","(OO)",arg1.getObject(), arg2.getObject());

    // set feature names
//    String featureNames = getYamlValue("PREDICTION/FEATURE_LIST");
//    CPyObject arg = Py_BuildValue("s",featureNames.c_str());
//    PyObject_CallMethod(predictor, "setFeatureNames","(O)",arg.getObject());
//
//    PyObject_CallMethod(predictor, "showFeatureCalculator",NULL);
}

Message
ForexTickClassifier::processMsg(Message& msg)
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
ForexTickClassifier::procMsg_CHECKIN(Message& msg)
{
    Log(LOG_INFO) << "Client checks in" <<std::endl;
    if ( !compareStringNoCase(m_fxSymbol, msg.getComment()) )
        Log(LOG_FATAL) << "Received symbol is inconsistent with model files" <<std::endl;

    Message msgnew;
    return msgnew;
}

Message
ForexTickClassifier::procMsg_HISTORY(Message& msg)
{
    Log(LOG_INFO) << "Msg of history data received" <<std::endl;

    int len = msg.getDataBytes()/sizeof(real64);
    CPyObject lst = PyList_New(len);
    real64* pm = (real64*)msg.getData();
    for ( int i = 0; i < len; i++ ) {
        PyList_SetItem(lst, i, Py_BuildValue("d", pm[i]));
    }
    if ( msg.getComment() == "buy" ) {
        PyObject_CallMethod(m_buyPredictor, (char*)"loadTicks","(O)",lst.getObject());
    } else if (msg.getComment() == "sell") {
        PyObject_CallMethod(m_sellPredictor, (char*)"loadTicks","(O)",lst.getObject());
    } else {
        Log(LOG_ERROR) << "Unexpected position type: " + msg.getComment() <<std::endl;
    }

    Log(LOG_INFO) << "History data loaded" <<std::endl;
    Message msgnew;
    return msgnew;
}

Message
ForexTickClassifier::procMsg_TICK(Message& msg)
{
    Log(LOG_DEBUG) << "New tick arrives" <<std::endl;
    real64 *pm = (real64*)msg.getData();
    real64 tick = pm[0];

    ActionType action;
    CPyObject pypred;
    CPyObject pytick = Py_BuildValue("d", tick);
    if ( msg.getComment() == "buy" ) {
        pypred = PyObject_CallMethod(m_buyPredictor, "classifyATick","(O)",pytick.getObject());
        action = (ActionType)FXAct::PLACE_BUY;
        Log(LOG_DEBUG) << "Buy tick arrives" <<std::endl;
    } else if (msg.getComment() == "sell") {
        pypred = PyObject_CallMethod(m_sellPredictor, "classifyATick","(O)",pytick.getObject());
        action = (ActionType)FXAct::PLACE_SELL;
        Log(LOG_DEBUG) << "Sell tick arrives" <<std::endl;
    } else {
        Log(LOG_ERROR) << "Unexpected position type: " + msg.getComment() <<std::endl;
        pypred = Py_BuildValue("i",1);
    }

    PyObject* objrepr = PyObject_Repr(pypred.getObject());
    if ( !objrepr )
        Log(LOG_ERROR) << "Failed to get prediction from python" <<std::endl;

    const char* cp = PyBytes_AsString(objrepr);
    cp = strdup(cp);

    Log(LOG_INFO) << "Prediction: " + String(cp) <<std::endl;
    int pred = stoi(String(cp));
    Py_XDECREF(objrepr);
    if ( pred == 1 )
        action = (ActionType)FXAct::NOACTION;
    else if(pred == 0)
        Log(LOG_INFO) << "Good to open position" <<std::endl;
        else
            Log(LOG_FATAL) << "Unrecognized prediction" <<std::endl;

    Message msgnew;
    msgnew.setAction(action);

    return msgnew;
}
