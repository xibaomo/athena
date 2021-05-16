/*
 * =====================================================================================
 *
 *       Filename:  client_base_app.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/04/2018 18:11:39
 *         Author:  fxua (), fxua@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "client_base_app.h"
#include "basics/utils.h"
#include "messenger/messenger.h"
using namespace std;
using namespace athena;
ClientBaseApp::ClientBaseApp() : App()
{
;
}

ClientBaseApp::~ClientBaseApp()
{
    shutdownServer();
}

void
ClientBaseApp::shutdownServer()
{
    Message msg;
    msg.setAction(MsgAct::NORMAL_EXIT);
    m_msger->sendAMsgToHostPort(msg, m_serverHostPort);
}

void
ClientBaseApp::wakeupServer(AppType atp)
{
    String localHp = " 127.0.0.1:" + to_string(m_msger->getPort());
    String flags = localHp + " " + to_string((int)atp);
    String athena_install = String(getenv("ATHENA_INSTALL"));
    String cmd = athena_install + "/api/release/scripts/run_api_server.sh";
    cmd = cmd + " " + flags + " &";

    system(cmd.c_str());

    Log(LOG_INFO) << "Launched api server. Waiting for response ...";

    Message msg;
    while ( m_msger->listenOnce(msg)>=0 ) {
        switch((MsgAct)msg.getAction()) {
            case MsgAct::GET_READY:
                break;
            case MsgAct::CHECK_IN:
                Log(LOG_INFO) << "Api server checked in";
                m_serverHostPort = msg.getComment();
                return;
                break;
            default:
                break;
        }
        msg.setAction(MsgAct::GET_READY);
        sleepMilliSec(ONE_MS);
    }
}

