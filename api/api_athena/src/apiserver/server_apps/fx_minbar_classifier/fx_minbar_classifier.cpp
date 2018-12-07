/*
 * =====================================================================================
 *
 *       Filename:  fx_minbar_classifier.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/06/2018 11:33:38 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "fx_minbar_classifier.h"
using namespace std;

void
ForexMinBarClassifier::prepare()
{
    loadPythonModule();

    String mf = getYamlValue("PREDICTION/BUY_MODEL");
    loadFilter(m_buyPredictor,mf);

    mf = getYamlValue("PREDICTION/SELL_MODEL");
    loadFilter(m_sellPredictor,mf);

    int lookback = stoi(getYamlValue("PREDICTION/LOOKBACK"));
    configPredictor(m_sellPredictor,lookback);
    configPredictor(m_buyPredictor,lookback);
}

void
ForexMinBarClassifier::configPredictor(CPyObject& predictor,int lookback)
{
    CPyObject arg = Py_BuildValue("i",lookback);
    PyObject_CallMethod(predictor,"setLookback","(O)",arg.getObject());


    // set feature names
    String featureNames = getYamlValue("PREDICTION/FEATURE_LIST");
    CPyObject ag = Py_BuildValue("s",featureNames.c_str());
    PyObject_CallMethod(predictor,"setFeatureNames","(O)",ag.getObject());
}

void
ForexMinBarClassifier::loadFilter(CPyObject& predictor, const String& modelFile)
{
    CPyObject arg = Py_BuildValue("s",modelFile.c_str());
    PyObject_CallMethod(predictor, "loadAModel","(O)",arg.getObject());

    Log(LOG_INFO) << "Model loaded: " + modelFile;
}

void
ForexMinBarClassifier::loadPythonModule()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/pyapi";
    m_pyInst.appendSysPath(modulePath);
    m_predictorModule = PyImport_ImportModule("forex_minbar_predictor");
    if (!m_predictorModule) {
        Log(LOG_FATAL) << "Failed to import module: forex_minbar_predictor";
    }

    CPyObject predictorClass = PyObject_GetAttrString(m_predictorModule,"ForexMinBarPredictor");
    if (!predictorClass) {
        Log(LOG_FATAL) << "Failed to get class: ForexMinBarPredictor";
    }

    m_buyPredictor = PyObject_CallObject(predictorClass,NULL);
    if (!m_buyPredictor) {
        Log(LOG_FATAL) << "Failed to create python buy predictor";
    }

    m_sellPredictor = PyObject_CallObject(predictorClass,NULL);
    if (!m_sellPredictor){
        Log(LOG_FATAL) << "Failed to create python sell predictor";
    }

}

Message
ForexMinBarClassifier::processMsg(Message& msg)
{
    Message msgnew;
    FXAction action = (FXAction)msg.getAction();
    switch(action) {
        case FXAction::TICK:
            msgnew = std::move(procMsg_MINBAR(msg));
            break;
        case FXAction::CHECKIN:
            msgnew = std::move(procMsg_CHECKIN(msg));
        default:
            break;
    }

    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_CHECKIN(Message& msg)
{
    Log(LOG_INFO) << "Client checks in";
    if ( !compareStringNoCase(m_fxSymbol, msg.getComment()) )
        Log(LOG_FATAL) << "Received symbol is inconsistent with model files";

    Message msgnew;
    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_MINBAR(Message& msg)
{
    Message msgnew;
    return msgnew;
}
