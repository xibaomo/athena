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

//    // load history file
//    CPyObject pyLatestTime;
//    ag = Py_BuildValue("s",histBarFile.c_str());
//    pyLatestTime = PyObject_CallMethod(predictor,"loadHistoryBarFile","(O)",ag.getObject());
//    if(!pyLatestTime) {
//        Log(LOG_FATAL) << "Failed to get latest time of bar file";
//    }
//
//    m_barFileLatestTime = getStringFromPyobject(pyLatestTime);

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
        case FXAction::MINBAR:
            msgnew = std::move(procMsg_MINBAR(msg));
            break;
        case FXAction::CHECKIN:
            msgnew = std::move(procMsg_CHECKIN(msg));
            break;
        case FXAction::HISTORY_MINBAR:
            msgnew = std::move(procMsg_HISTORY_MINBAR(msg));
            break;
        case FXAction::INIT_TIME:
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
    Log(LOG_INFO) << "Client checks in";
    if ( !compareStringNoCase(m_fxSymbol, msg.getComment()) )
        Log(LOG_FATAL) << "Received symbol is inconsistent with model files";

    Message msgnew;
    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_MINBAR(Message& msg)
{
    Log(LOG_INFO) << "New min bar arrives:";
    Real* pm = (Real*)msg.getData();

    Log(LOG_INFO) << to_string(pm[0]) << " "
                  << to_string(pm[1]) << " "
                  << to_string(pm[2]) << " "
                  << to_string(pm[3]) << " "
                  << to_string(pm[4]) << " ";

    ActionType action;
    CPyObject pypred;
    CPyObject pyopen = Py_BuildValue(REALFORMAT,pm[0]);
    CPyObject pyhigh = Py_BuildValue(REALFORMAT,pm[1]);
    CPyObject pylow  = Py_BuildValue(REALFORMAT,pm[2]);
    CPyObject pyclose= Py_BuildValue(REALFORMAT,pm[3]);
    CPyObject pytickvol=Py_BuildValue(REALFORMAT,pm[4]);

    pypred = PyObject_CallMethod(m_buyPredictor,"classifyMinBar","(OOOOO)",pyopen.getObject(),
                                 pyhigh.getObject(),
                                 pylow.getObject(),
                                 pyclose.getObject(),
                                 pytickvol.getObject());

    if (!pypred) {
        Log(LOG_FATAL) << "Buy prediction failed";
    }
    int buy_pred = getIntFromPyobject(pypred);

//    pypred = PyObject_CallMethod(m_sellPredictor,"classifyMinBar","(OOOOO)",pyopen.getObject(),
//                                 pyhigh.getObject(),
//                                 pylow.getObject(),
//                                 pyclose.getObject(),
//                                 pytickvol.getObject());
//
//    if(!pypred) {
//        Log(LOG_FATAL) << "Sell prediction failed";
//    }
//    int sell_pred = getIntFromPyobject(pypred);

//    if (buy_pred == 0 && sell_pred == 0) {
//        Log(LOG_WARNING) << "Good to place both positions. Too risky, no action";
//        action = (ActionType)FXAction::NOACTION;
//    } else if (buy_pred == 0 && sell_pred == 1) {
//        action = (ActionType)FXAction::PLACE_BUY;
//        Log(LOG_INFO) << "Place buy position";
//    } else if (buy_pred == 1 && sell_pred == 0) {
//        action = (ActionType)FXAction::PLACE_SELL;
//        Log(LOG_INFO) << "Place sell position";
//    } else {
//        action = (ActionType)FXAction::NOACTION;
//        Log(LOG_INFO) << "Bad for either position. No action.";
//    }

//    if (action == (ActionType)FXAction::PLACE_SELL )
//        action =  (ActionType)FXAction::NOACTION;
    if (buy_pred == 0) {
        action = (ActionType)FXAction::PLACE_BUY;
        Log(LOG_INFO) << "Place buy position";
    } else {
        action = (ActionType)FXAction::NOACTION;
        Log(LOG_INFO) << "No action";
    }
    Message msgnew;
    msgnew.setAction(action);

    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_HISTORY_MINBAR(Message& msg)
{
    int *pc = (int*)msg.getChar();
    int histLen = pc[0];
    int minbar_size = pc[1];
    Real* pm = (Real*)msg.getData();
    CPyObject lst = PyList_New(histLen*minbar_size);
    for (int i=0;i < histLen*minbar_size; i++) {
        PyList_SetItem(lst,i,Py_BuildValue(REALFORMAT,pm[i]));
    }
    CPyObject pylookback = Py_BuildValue("i",histLen);
    CPyObject pyminbarsize = Py_BuildValue("i",minbar_size);
    PyObject_CallMethod(m_buyPredictor,"loadHistoryMinBars","(OOO)",
                        lst.getObject(),
                        pylookback.getObject(),
                        pyminbarsize.getObject());
    Log(LOG_INFO) << "Buy predictor loads history min bars. History length: " + to_string(histLen);

    PyObject_CallMethod(m_sellPredictor,"loadHistoryMinBars","(OOO)",
                        lst.getObject(),
                        pylookback.getObject(),
                        pyminbarsize.getObject());

    Log(LOG_INFO) << "Sell predictor loads history min bars. History length: " + to_string(histLen);
    Message msgnew;
    return msgnew;
}

Message
ForexMinBarClassifier::procMsg_INIT_TIME(Message& msg)
{
    Message msgnew;
    String initTime = msg.getComment();
    Log(LOG_INFO) << "MT5 starting time: " + initTime;

    String latestMinBar;
    CPyObject pyLatestMinbar;

    String barFile = getYamlValue("PREDICTION/HISTORY_BAR_FILE");
    CPyObject pyBarFile = Py_BuildValue("s",barFile.c_str());
    CPyObject pyInitMin = Py_BuildValue("s",initTime.c_str());
    PyObject_CallMethod(m_buyPredictor,"setInitMin","(O)",pyInitMin.getObject());
    pyLatestMinbar = PyObject_CallMethod(m_buyPredictor,"loadHistoryBarFile","(O)",pyBarFile.getObject());

    PyObject_CallMethod(m_sellPredictor,"setInitMin","(O)",pyInitMin.getObject());
    pyLatestMinbar = PyObject_CallMethod(m_sellPredictor,"loadHistoryBarFile","(O)",pyBarFile.getObject());

    latestMinBar = getStringFromPyobject(pyLatestMinbar);

    Log(LOG_INFO) << "Latest min bar in history: " + latestMinBar;
    auto diffTime = getTimeDiffInMin(initTime,latestMinBar);

    int histLen;
    if (diffTime>0) {
        histLen = diffTime;
    } else {
        histLen = 0;
    }
    Message m(sizeof(int),0);
    msgnew = std::move(m);
    int* pm = (int*)msgnew.getData();
    pm[0] = histLen;
    msgnew.setAction((ActionType)FXAction::REQUEST_HISTORY_MINBAR);
    Log(LOG_INFO) << "Request client to send history min bars: " + to_string(histLen);

    return msgnew;
}
