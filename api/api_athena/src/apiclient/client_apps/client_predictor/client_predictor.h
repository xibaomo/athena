/*
 * =====================================================================================
 *
 *       Filename:  client_predictor.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/04/2018 18:23:55
 *
 *         Author:  fxua (), fxua@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#ifndef  _CLIENT_APP_PREDICTOR_H_
#define  _CLIENT_APP_PREDICTOR_H_
#include "client_apps/client_base_app/client_base_app.h"

class ClientPredictor: public ClientBaseApp {
public:
    ClientPredictor() {;}
    Real* m_resultArray;
public:
    virtual ~ClientPredictor() {;}

    static ClientPredictor& getInstance() {
        static ClientPredictor _ins;
        return _ins;
    }

/*
 * Preparation
 * wake up api server and send model file
 */
    void prepare(const String& model_file);

/*
 * Send engine config to api server
 */
    void sendEngineConfig(const String& modelfile);

/*
 * Send feature matrix to api server
 */
    void sendFeatures(Real* fm, const Uint rows, const Uint cols);

/*
 * Wait until received message from api server
 */
    void waitResult();

/*
 * Process the result message from api server
 */
    void procResultMsg(Message& msg);

/*
 * Copy result to user's container
 * 'out' must be allocated beforehand by user
 */
    void getResult(Real* out, int len);
    void execute() {;}

    void finish() {;}
};

#endif   /* ----- #ifndef _CLIENT_APP_PREDICTOR_H_  ----- */
