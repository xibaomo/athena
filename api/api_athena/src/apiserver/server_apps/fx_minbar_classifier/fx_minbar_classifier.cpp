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
using namespace athena;
void
ForexMinBarClassifier::prepare()
{
    loadPythonModule();

    //String mf = getYamlValue("PREDICTION/BUY_MODEL");
    //loadFilter(m_buyPredictor,mf);

//    int lookback = stoi(getYamlValue("PREDICTION/LOOKBACK"));
//    int malookback = stoi(getYamlValue("PREDICTION/MA_LOOKBACK"));
//    configPredictor(m_buyPredictor,lookback,malookback);

//    String barFile = getYamlValue("PREDICTION/HISTORY_BAR_FILE");
//    m_histBarFile = barFile;
//    CPyObject pyBarFile = Py_BuildValue("s",barFile.c_str());
//    m_pyLatestMinbar = PyObject_CallMethod(m_buyPredictor,"loadHistoryBarFile","(O)",pyBarFile.getObject());

    Log(LOG_INFO) << "First loading history file, latest bar time: " + getStringFromPyobject(m_pyLatestMinbar) <<std::endl;

}

void
ForexMinBarClassifier::configPredictor(CPyObject& predictor,int lookback, int malookback)
{
    CPyObject arg = Py_BuildValue("i",lookback);
    PyObject_CallMethod(predictor,"setLookback","(O)",arg.getObject());

    CPyObject arg1 = Py_BuildValue("i",malookback);
    PyObject_CallMethod(predictor,"setMALookback","(O)",arg1.getObject());

    // set feature names
//    String featureNames = getYamlValue("PREDICTION/FEATURE_LIST");
//    CPyObject ag = Py_BuildValue("s",featureNames.c_str());
//    PyObject_CallMethod(predictor,"setFeatureNames","(O)",ag.getObject());

}

void
ForexMinBarClassifier::loadFilter(CPyObject& predictor, const String& modelFile)
{
    CPyObject arg = Py_BuildValue("s",modelFile.c_str());
    PyObject_CallMethod(predictor, "loadAModel","(O)",arg.getObject());

    Log(LOG_INFO) << "Model loaded: " + modelFile <<std::endl;
}

void
ForexMinBarClassifier::loadPythonModule()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/pyapi";
    PyEnviron::getInstance().appendSysPath(modulePath);
    m_predictorModule = PyImport_ImportModule("forex_minbar_predictor");
    if (!m_predictorModule) {
        Log(LOG_FATAL) << "Failed to import module: forex_minbar_predictor" <<std::endl;
    }

    CPyObject predictorClass = PyObject_GetAttrString(m_predictorModule,"ForexMinBarPredictor");
    if (!predictorClass) {
        Log(LOG_FATAL) << "Failed to get class: ForexMinBarPredictor" <<std::endl;
    }

    m_buyPredictor = PyObject_CallObject(predictorClass,NULL);
    if (!m_buyPredictor) {
        Log(LOG_FATAL) << "Failed to create python buy predictor" <<std::endl;
    }

}

Message
ForexMinBarClassifier::processMsg(Message& msg)
{
    Message msgnew;
    FXAct action = (FXAct)msg.getAction();
    switch(action) {
    case FXAct::NEW_MINBAR:
        msgnew = std::move(procMsg_MINBAR(msg));
        break;
    case FXAct::HISTORY_MINBAR:
        msgnew = std::move(procMsg_HISTORY_MINBAR(msg));
        break;
    case FXAct::INIT_TIME:
        msgnew = std::move(procMsg_INIT_TIME(msg));
        break;
    default:
        break;
    }

    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_CHECKIN(Message& msg)
{
    Log(LOG_INFO) << "Client checks in" <<std::endl;
    if ( !compareStringNoCase(m_fxSymbol, msg.getComment()) ) {
        Log(LOG_FATAL) << "Received symbol is inconsistent with model files" <<std::endl;
    }

    Message msgnew;
    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_MINBAR(Message& msg)
{
    Log(LOG_INFO) << "New min bar arrives: " + msg.getComment() + " + 00:05 " <<std::endl;
    real64* pm = (real64*)msg.getData();

    Log(LOG_INFO) << to_string(pm[0]) + " "
                  + to_string(pm[1]) + " "
                  + to_string(pm[2]) + " "
                  + to_string(pm[3]) + " "
                  + to_string(pm[4]) + " ";

    ActionType action;
    CPyObject pypred;
    CPyObject pyopen = Py_BuildValue("d",pm[0]);
    CPyObject pyhigh = Py_BuildValue("d",pm[1]);
    CPyObject pylow  = Py_BuildValue("d",pm[2]);
    CPyObject pyclose= Py_BuildValue("d",pm[3]);
    CPyObject pytickvol=Py_BuildValue("d",pm[4]);
    CPyObject pytime = Py_BuildValue("s",msg.getComment().c_str());

    pypred = PyObject_CallMethod(m_buyPredictor,"classifyMinBar","(OOOOOO)",pyopen.getObject(),
                                 pyhigh.getObject(),
                                 pylow.getObject(),
                                 pyclose.getObject(),
                                 pytickvol.getObject(),
                                 pytime.getObject());

    if (!pypred) {
        Log(LOG_FATAL) << "Buy prediction failed" <<std::endl;
    }
    int buy_pred = getIntFromPyobject(pypred);

    if (buy_pred == 0) {
        action = (ActionType)FXAct::PLACE_BUY;
        Log(LOG_INFO) << "Place buy position" <<std::endl;
    } else {
        action = (ActionType)FXAct::NOACTION;
        Log(LOG_INFO) << "No action" <<std::endl;
    }
    Message msgnew;
    msgnew.setAction(action);

    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_HISTORY_MINBAR(Message& msg)
{
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    int histLen = pack.int32_vec[0];

    if (histLen >0) {
        int minbar_size = pack.int32_vec[1];
        real64* pm = &pack.real64_vec[0];
        CPyObject lst = PyList_New(histLen*minbar_size);
        for (int i=0; i < histLen*minbar_size; i++) {
            PyList_SetItem(lst,i,Py_BuildValue("d",pm[i]));
        }
        CPyObject pylookback = Py_BuildValue("i",histLen);
        CPyObject pyminbarsize = Py_BuildValue("i",minbar_size);
        PyObject_CallMethod(m_buyPredictor,"loadHistoryMinBars","(OOO)",
                            lst.getObject(),
                            pylookback.getObject(),
                            pyminbarsize.getObject());
        Log(LOG_INFO) << "Buy predictor loads history min bars. History length: " + to_string(histLen) <<std::endl;
    } else {
        Log(LOG_INFO) << "No min bars from mt5, predicting unlabeled bars in history ..." <<std::endl;
        PyObject_CallMethod(m_buyPredictor,"predictHistoryMinBars",NULL);

        Log(LOG_INFO) << "All history bars are labeled" <<std::endl;
    }


    Message msgnew;
    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_INIT_TIME(Message& msg)
{
    String initTime = msg.getComment();
    Log(LOG_INFO) << "MT5 latest bar: " + initTime <<std::endl;

    String latestMinBar;
    CPyObject pyLatestMinbar;


    CPyObject pyInitMin = Py_BuildValue("s",initTime.c_str());
    CPyObject pyBarfile = Py_BuildValue("s",m_histBarFile.c_str());
    PyObject_CallMethod(m_buyPredictor,"setInitMin","(OO)",pyInitMin.getObject(),pyBarfile.getObject());

    latestMinBar = getStringFromPyobject(m_pyLatestMinbar);

    auto diffTime = getTimeDiffInMin(initTime,latestMinBar);

    int histLen;
    if (diffTime>0) {
        histLen = diffTime;
    } else {
        histLen = 0;
    }

//    latestMinBar = convertTimeString(latestMinBar,"%Y.%m.%d %H:%M");
//    Log(LOG_INFO) << "Converted latest min bar: " + latestMinBar <<std::endl;

    Message msgnew(sizeof(int),latestMinBar.size());
    msgnew.setComment(latestMinBar);
    int* pm = (int*)msgnew.getData();
    pm[0] = histLen;
    msgnew.setAction(FXAct::REQUEST_HISTORY_MINBAR);
    Log(LOG_INFO) << "Request client to send history min bars: " + to_string(histLen) <<std::endl;

    return msgnew;
}
