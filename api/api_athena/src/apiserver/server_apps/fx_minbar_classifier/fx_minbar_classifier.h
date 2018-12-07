/*
 * =====================================================================================
 *
 *       Filename:  fx_minbar_classifier.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/06/2018 11:33:01 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _FX_MINBAR_CLASSIFIER_H_
#define  _FX_MINBAR_CLASSIFIER_H_

#include "server_apps/fx_tick_classifier/fx_action.h"
#include "server_apps/server_base_app/server_base_app.h"
#include "pyhelper.hpp"
#include "basics/utils.h"
class ForexMinBarClassifier : public ServerBaseApp {
protected:
    CPyInstance m_pyInst;
    CPyObject   m_buyPredictor;
    CPyObject   m_sellPredictor;
    CPyObject   m_predictorModule;

    String      m_fxSymbol;

    ForexMinBarClassifier(const String& configFile) : ServerBaseApp(configFile) {
        m_fxSymbol =  getYamlValue("PREDICTION/SYMBOL");
        Log(LOG_INFO) << "Forex minbar classifier created. Symbol: " + m_fxSymbol;
    }
public:
    virtual ~ForexMinBarClassifier() {;}
    static ForexMinBarClassifier& getInstance(const String& cf) {
        static ForexMinBarClassifier _instance(cf);
        return _instance;
    }

    void prepare();

    void finish() {;}

    void loadPythonModule();
    void loadFilter(CPyObject& predictor, const String& modelFile);

    /**
     * Config predictor
     */
    void configPredictor(CPyObject& predictor, int lookback);

    /**
     * Process incoming msg, return response msg
     */
    Message processMsg(Message& msg);

    /**
     * Process msg with various actions
     */
    Message procMsg_CHECKIN(Message& msg);
    Message procMsg_MINBAR(Message& msg);
};
#endif   /* ----- #ifndef _FX_MINBAR_CLASSIFIER_H_  ----- */
