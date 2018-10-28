#include <iostream>
#include "basics/log.h"
#include "basics/utils.h"
#include "basics/types.h"
#include "predictor/prdmsg.h"
#include "messenger/messenger.h"
using namespace std;

int main(int argc, char** argv)
{
    Log.setLogLevel(LOG_INFO);

    Log(LOG_INFO) << "Program starts";

    if (argc == 1) { // main process
        Log(LOG_INFO) << "This is master process";

        Messenger* msger = &Messenger::getInstance();
        String cmd = String(argv[0])+ " 127.0.0.1:" + to_string(msger->getPort());
//        system((cmd + "&").c_str());
        NonBlockSysCall nbcall(cmd);
        Log(LOG_INFO) << "another guy wakes up";
        Message msg;
        while(msger->listenOnce(msg)>=0) {
            switch((MsgAction)msg.getAction()) {
            case MsgAction::GET_READY:
                break;
            case MsgAction::CHECK_IN:
                Log(LOG_INFO) << "another guy is ready";
                return 0;
                break;
            default:
                break;

            }
            msg.setAction(MsgAction::GET_READY);
            sleepMilliSec(ONE_MS);
        }


    } else {
        Log(LOG_INFO) << "This is second process";
        String hostPort(argv[1]);
        Messenger* msger =  &Messenger::getInstance();

        Message msg;
        msg.setAction(MsgAction::CHECK_IN);

        msger->sendAMsgToHostPort(msg,hostPort);


    }


    return 0;
}
