/*
 * =====================================================================================
 *
 *       Filename:  server_base_app.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/14/2018 01:36:53 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#include "server_apps/server_base_app/server_base_app.h"
#include "messenger/msg.h"
#include "basics/utils.h"
#include "messenger/sockutils.h"

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

//void
//ServerBaseApp::_execute()
//{
//    Message msg;
//    while ( m_msger->listenOnce(msg) >=0 ) {
//        MsgAction action = (MsgAction) msg.getAction();
//        if ( action == MsgAction::GET_READY ) {
//            sleepMilliSec(ONE_MS);
//            continue;
//        }
//        processMsg(msg); // Implemented by concrete class
//        msg.setAction(MsgAction::GET_READY);
//    }
//}

void
ServerBaseApp::execute()
{
    char buf[16];
    while(1) {
        int sock = m_msger->getHostSocket();
        bool rt = checkSockReadable(sock,1);
        if (rt) {
            int clntsock;
            if ((clntsock = accept(sock,NULL,NULL)) <0)
                Log(LOG_FATAL) << "Failed to accept connection";
            m_msger->drainSocket(clntsock);
            Message msg = m_msger->popMsgBox();
            if (msg.getMsgSize() == 0)
                return;
            MsgAction action = (MsgAction)msg.getAction();
            if (action == MsgAction::NORMAL_EXIT)
                return;

            Message msgReply = std::move(processMsg(msg));
            if ((MsgAction)msgReply.getAction() == MsgAction::GET_READY) {
                m_msger->shutdownConnection(clntsock);
                close(clntsock);
                continue;
            }
            m_msger->writeSocket(clntsock,msgReply);

            int res = recv(clntsock,buf,16,0);
            if (res == 0) {
                close(clntsock);
            } else {
                Log(LOG_FATAL) << "Unexpected msg received";
            }
        }
        sleepMilliSec(ONE_MS);
    }
}
