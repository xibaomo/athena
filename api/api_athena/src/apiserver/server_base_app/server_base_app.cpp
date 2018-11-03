//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
#include "server_apps/server_base_app/server_base_app.h"
#include "messenger/msg.h"
#include "basics/utils.h"

using namespace std;

ServerBaseApp::ServerBaseApp(const String& clientHostPort):
                        m_clientHostPort(clientHostPort)
{
    String cmt = "127.0.0.1:" + to_string(m_msger->getPort());
    Message msg(0, cmt.size());
    msg.setComment(cmt);
    msg.setAction(MsgAction::CHECK_IN);

    m_msger->sendAMsgToHostPort(msg, clientHostPort);
    Log(LOG_INFO) << "Msg sent to client: " + clientHostPort;
}
void
ServerBaseApp::execute()
{
    Message msg;
    while ( m_msger->listenOnce(msg) >=0 ) {
        MsgAction action = (MsgAction) msg.getAction();
        if ( action == MsgAction::GET_READY ) {
            sleepMilliSec(ONE_MS);
            continue;
        }
        processMsg(msg); // Implemented by concrete class
        msg.setAction(MsgAction::GET_READY);
    }
}

