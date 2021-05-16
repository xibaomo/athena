/*
 * =====================================================================================
 *
 *       Filename:  client_predictor.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/04/2018 18:30:32
 *
 *         Author:  fxua (), fxua@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "client_predictor.h"
#include "predictor/prdtypes.h"
#include "basics/utils.h"
using namespace std;
using namespace athena;
void
ClientPredictor::prepare(const String& modelfile)
{
    wakeupServer(AppType::APP_PREDICTOR);
    sendEngineConfig(modelfile);
}

void
ClientPredictor::sendEngineConfig(const String& modelfile)
{
    int dataBytes = sizeof(int)*2;
    int charBytes = modelfile.size();
    Message msg(dataBytes, charBytes);
    msg.setAction(PrdAction::SETUP);
    int *pm = (int*)msg.getData();
    pm[0] = (int)EngineType::REGRESSOR;
    pm[1] = (int)EngineCoreType::RANDOMFOREST;
    msg.setComment(modelfile);
    m_msger->sendAMsgToHostPort(msg, m_serverHostPort);

    Log(LOG_INFO) << "Model file sent to api server: " + modelfile;
}

void
ClientPredictor::sendFeatures(Real* fm, const Uint rows, const Uint cols)
{
    m_resultArray = new Real[rows];
    int dataBytes = sizeof(Real) * rows * cols;
    int charBytes = sizeof(Uint) * 2;
    Message msg(dataBytes, charBytes);
    msg.setAction(PrdAction::PREDICT);
    void* pm = (void*)msg.getData();
    memcpy(pm, (void*)fm, dataBytes);
    Uint* pc = (Uint*)msg.getChar();
    pc[0] = rows;
    pc[1] = cols;

    m_msger->sendAMsgToHostPort(msg, m_serverHostPort);

    waitResult();

    Log(LOG_VERBOSE) << "Prediction done";
}

void
ClientPredictor::waitResult()
{
    Message msg;
    while ( m_msger->listenOnce(msg)>=0 ) {
        if ( (MsgAct)msg.getAction() == MsgAct::GET_READY ) {
            sleepMilliSec(ONE_MS);
            continue;
        }
        PrdAction action = (PrdAction)msg.getAction();
        switch(action) {
            case PrdAction::RESULT:
                procResultMsg(msg);
                return;
                break;
            default:
                break;
        }
        msg.setAction(MsgAct::GET_READY);
        sleepMilliSec(ONE_MS);
    }
}

void
ClientPredictor::procResultMsg(Message& msg)
{
    Real* pm = (Real*)msg.getData();
    memmove((void*)m_resultArray, pm, msg.getDataBytes());
}

void
ClientPredictor::getResult(Real* out, int len)
{
    memmove(out, m_resultArray, sizeof(Real)*len);
    delete m_resultArray;
}
