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
#include "predictor/prdtypes.h"

class ServerPredictor : public ServerBaseApp
{
protected:

    CPyInstance m_pyInst; // Initialize Python environment. must be the first. last destroyed
    CPyObject m_engine;
    CPyObject m_engineCore;

    CPyObject m_engCreatorMod; // module for engine creator
    CPyObject m_mlEngMod;     // module for mlengine base
    CPyObject m_engineCoreMod; // module for mlengine core
    double *m_result_array;

    ServerPredictor(const String& clientHostPort): ServerBaseApp(clientHostPort),
    m_result_array(nullptr){;}
public:
    virtual ~ServerPredictor() {;}

    static ServerPredictor& getInstance(const String& hostPort)
    {
        static ServerPredictor _instance(hostPort);
        return _instance;
    }

    /*
     * Load python Module
     */
    void loadPythonModule();

    /*
     * Load model file and create ML engine & core
     */
    void loadEngine(EngineType et, EngineCoreType ect, const String& modelfile);

    /*
     * Predict upon input feature matrix
     */
    void predict(Real* fm, const Uint n_samples, const Uint n_features);

    /*
     * Preparation:
     * 1. load python modules
     */
    void prepare();

    /*
     * Process message. Override base class
     */
    Message processMsg(Message& msg);

    /*
     * Process message of action SETUP
     * Build up ML engine
     */
    void procMsg_SETUP(Message& msg);

    /*
     * Process message of action PREDICT
     * Make prediction with ML engine
     * Return results to client
     */
    void procMsg_PREDICT(Message& msg);
};

#endif   /* ----- #ifndef _SERVER_PREDICTOR_H_  ----- */
