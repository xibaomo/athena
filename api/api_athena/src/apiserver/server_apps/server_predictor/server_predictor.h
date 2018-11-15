/*
 * =====================================================================================
 *
 *       Filename:  server_predictor.h
 *
 *    Description:  j
 *
 *        Version:  1.0
 *        Created:  11/04/2018 17:14:43
 *         Author:  fxua (), fxua@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _SERVER_PREDICTOR_H_
#define  _SERVER_PREDICTOR_H_

#include "server_apps/server_base_app/server_base_app.h"
#include "pyhelper.hpp"
#include "basics/utils.h"
#include <vector>
class ServerPredictor : public ServerBaseApp
{
protected:

    CPyInstance m_pyInst; // Initialize Python environment. must be the first. last destroyed
    CPyObject m_engine;
    CPyObject m_engineCore;
    std::vector<CPyObject> m_buyEngines;
    std::vector<CPyObject> m_sellEngines;

    CPyObject m_engCreatorMod; // module for engine creator
    CPyObject m_mlEngMod;     // module for mlengine base
    CPyObject m_engineCoreMod; // module for mlengine core
    CPyObject m_overkillFilterMod;
    double *m_result_array;

    String m_yamlParser;
    String m_fxSymbol;

    ServerPredictor(const String& config): ServerBaseApp(config),
    m_result_array(nullptr){
        m_yamlParser = String(getenv("ATHENA_INSTALL")) + "/api/release/scripts/yaml_parser.py ";
    }
public:
    virtual ~ServerPredictor() {;}

    static ServerPredictor& getInstance(const String& config)
    {
        static ServerPredictor _instance(config);
        return _instance;
    }

    String getYamlValue(const String& key)
    {
        String cmd = m_yamlParser + key + " " + m_configFile;
        String val = execSysCall_block(cmd);
        return val;
    }

    /**
     * Load  Models from config file
     */
    void loadAllFilters();

    /**
     * Load a filter set.
     * pos_type: "buy" or "sell"
     */
    void loadFilterSet(const String& symbol, const String& pos_type);

    /**
     * Load python Module
     */
    void loadPythonModule();

    /**
     * Load model file and create ML engine & core
     */
    void loadEngine(EngineType et, EngineCoreType ect, const String& modelfile);

    /**
     * Predict upon input feature matrix
     */
    void predict(Real* fm, const Uint n_samples, const Uint n_features);

    /**
     * Preparation:
     * 1. load python modules
     */
    void prepare();

    /**
     * Process message. Override base class
     */
    Message processMsg(Message& msg);

    /**
     * Process message of action SETUP
     * Build up ML engine
     */
    void procMsg_HISTORY(Message& msg);

    /**
     * Process message of action PREDICT
     * Make prediction with ML engine
     * Return results to client
     */
    void procMsg_TICK(Message& msg);
};

#endif   /* ----- #ifndef _SERVER_PREDICTOR_H_  ----- */
