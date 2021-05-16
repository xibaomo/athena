/*
 * =====================================================================================
 *
 *       Filename:  fxclassifier.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/16/2018 01:49:27 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _FOREX_TICK_CLASSIFIER_H_
#define  _FOREX_TICK_CLASSIFIER_H_

#include "server_apps/server_base_app/server_base_app.h"
#include "pyhelper.hpp"
#include "basics/utils.h"
#include "fx_action/fx_action.h"
class ForexTickClassifier : public ServerBaseApp {
protected:
    CPyObject   m_buyPredictor;
    CPyObject   m_sellPredictor;

    CPyObject   m_predictorModule;

    String      m_fxSymbol;

    ForexTickClassifier(const String& configFile) : ServerBaseApp(configFile) {
        m_fxSymbol = getYamlValue("PREDICTION/SYMBOL");
        Log(LOG_INFO) << "Forex classifier created. Symbol: " + m_fxSymbol;
    }
public:
    virtual ~ForexTickClassifier() {;}
    static ForexTickClassifier& getInstance(const String& cf) {
        static ForexTickClassifier _inst(cf);
        return _inst;
    }

    void prepare();
    void finish() {;}

    void loadPythonModules();
    void loadFilter(CPyObject& pred, const String& modelFile);

    /**
     * Config predictor
     */
    void configPredictor(CPyObject& pred, int fastperiod, int slowperiod);

    /**
     * Process incoming msg, return response msg
     */
    Message processMsg(Message& msg);

    /**
     * Process msg with various actions
     */
    Message procMsg_CHECKIN(Message& msg);
    Message procMsg_HISTORY(Message& msg);
    Message procMsg_TICK(Message& msg);
};
#endif   /* ----- #ifndef _FOREX_TICK_CLASSIFIER_H_  ----- */
