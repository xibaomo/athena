#include "minbar_tracker.h"
#include "fx_action/fx_action.h"
using namespace std;
using namespace athena;

Message
MinBarTracker::processMsg(Message& msg)
{
    Message msgnew;
    FXAction action = (FXAction)msg.getAction();
//    switch(action) {
//    case FXAction::MINBAR:
//        msgnew = std::move(procMsg_MINBAR(msg));
//        break;
//    case FXAction::CHECKIN:
//        msgnew = procMsg_noreply(msg,[this](Message& m){
//                                 Log(LOG_INFO) << "Client checked in";
//                                 })
//        break;
//    case FXAction::HISTORY_MINBAR:
//        msgnew = std::move(procMsg_HISTORY_MINBAR(msg));
//        break;
//    case FXAction::INIT_TIME:
//        msgnew = std::move(procMsg_INIT_TIME(msg));
//        break;
//    default:
//        break;
//    }

    return msgnew;
}
