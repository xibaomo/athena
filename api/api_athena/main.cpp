#include <iostream>
#include "basics/log.h"
#include "basics/utils.h"
#include "basics/types.h"
#include "messenger/messenger.h"
using namespace std;
using namespace athena;
int main(int argc, char** argv)
{
    Log::setLogLevel(LOG_INFO);

    Log(LOG_INFO) << "Program starts";

    Log(LOG_INFO) << "test";

//    if ( argc == 1) { // main process
        Log(LOG_INFO) << "This is master process";
        Log(LOG_INFO) << "test2";

        Messenger* msger = &Messenger::getInstance();
        String cmd = String(argv[0])+ " 127.0.0.1:" + to_string(msger->getPort());
//        system((cmd + "&").c_str());
//        system(cmd.c_str());
        Log(LOG_INFO) << "another guy wakes up";
        return 0;
        Message msg;
        while ( msger->listenOnce(msg)>=0 ) {
            switch((MsgAct)msg.getAction()) {
            case MsgAct::GET_READY:
                break;
            case MsgAct::CHECK_IN:
                Log(LOG_INFO) << "another guy is ready";
                return 0;
                break;
            default:
                break;
            }
            msg.setAction(MsgAct::GET_READY);
            sleepMilliSec(ONE_MS);
        }
//    } else {
//        Log(LOG_INFO) << "This is second process";
//        String hostPort(argv[1]);
//        Messenger* msger =  &Messenger::getInstance();
//
//        Message msg;
//        msg.setAction(MsgAct::CHECK_IN);
//
//        msger->sendAMsgToHostPort(msg, hostPort);
//    }

//    return 0;
}
