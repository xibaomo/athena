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
using namespace athena;

void
ServerBaseApp::execute() {
    char buf[16];
    Timer timer;
    int prev_time_point=0;
    Log(LOG_INFO) << "Listening to port: " + to_string(m_msger->getPort()) <<std::endl;
    while(1) {
        int sock = m_msger->getHostSocket();
        bool rt = checkSockReadable(sock,1);
        if (rt) {
            int clntsock;
            if ((clntsock = accept(sock,NULL,NULL)) <0)
                Log(LOG_FATAL) << "Failed to accept connection" <<std::endl;
            Log(LOG_DEBUG) << "New connection comes in" <<std::endl;
            m_msger->drainSocket(clntsock);
            Message msg = m_msger->popMsgBox();
            if (msg.getMsgSize() == 0)
                return;
            MsgAct action = (MsgAct)msg.getAction();
            if (action == MsgAct::NORMAL_EXIT)
                return;

            Message msgReply;
            if (action == MsgAct::CHECK_IN) {
                msgReply = procMsg_noreply(msg,[this](const Message& msg) {
                    Log(LOG_INFO) << "Client checked in" <<std::endl;
                });
            } else {
                msgReply = std::move(processMsg(msg));
            }

            if ((MsgAct)msgReply.getAction() == MsgAct::GET_READY) {
                Log(LOG_DEBUG) << "No reply, shut down connection" <<std::endl;
                m_msger->shutdownConnection(clntsock);
                close(clntsock);
                continue;
            }
            m_msger->writeSocket(clntsock,msgReply);

            int res = recv(clntsock,buf,16,0);
            if (res == 0) {
                close(clntsock);
            } else {
                Log(LOG_FATAL) << "Unexpected msg received" <<std::endl;
            }
        }
        sleepMilliSec(ONE_MS);

        int elapsed = timer.getElapsedTime();

        if (elapsed > prev_time_point && elapsed % 60 == 0) {
            prev_time_point = elapsed;
            Log(LOG_INFO) << "Listening to port: " + to_string(m_msger->getPort()) <<std::endl;
        }
    }
}
