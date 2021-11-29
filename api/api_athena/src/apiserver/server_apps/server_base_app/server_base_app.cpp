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
    Timer timer;
    int prev_time_point=0;
    Log(LOG_INFO) << "Listening to port: " + to_string(m_msger->getPort()) <<std::endl;

    while(1) {
        Message msg;
        int rt = m_msger->listenOnce(msg);
        if (rt < 0) break;
        if (rt == 0) continue;

        Message msgReply;
        MsgAct action = (MsgAct)msg.getAction();
        if (action == MsgAct::CHECK_IN) {
            msgReply = procMsg_noreply(msg,[this](const Message& msg) {
                Log(LOG_INFO) << "Client checked in" <<std::endl;
            });
        } else {
            msgReply = std::move(processMsg(msg));
        }

        if ((MsgAct)msgReply.getAction() == MsgAct::GET_READY) {
            Log(LOG_DEBUG) << "No reply" <<std::endl;
            continue;
        }
        m_msger->sendToPartner(msgReply);

        sleepMilliSec(ONE_MS);

        int elapsed = timer.getElapsedTime();

        if (elapsed > prev_time_point && elapsed % 60 == 0) {
            prev_time_point = elapsed;
            Log(LOG_INFO) << "Listening to port: " + to_string(m_msger->getPort()) <<std::endl;
        }
    }
}
