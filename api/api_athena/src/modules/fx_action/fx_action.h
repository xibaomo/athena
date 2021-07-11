#ifndef _CLIENT_API_FX_ACTION_H_
#define _CLIENT_API_FX_ACTION_H_
#include "messenger/msg.h"

enum class FXAct {
    HISTORY = MsgAct::NUM_ACTIONS,
    HISTORY_MINBAR,
    //CHECKIN,
    TICK,
    MINBAR,
    NOACTION,
    PLACE_BUY,
    PLACE_SELL,
    INIT_TIME,
    REQUEST_HISTORY_MINBAR,
    PROFIT,
    CLOSE_POS,
    ASK_PAIR,
    PAIR_HIST_X,
    PAIR_HIST_Y,
    PAIR_MIN_OPEN,
    CLOSE_ALL_POS,
    SYM_HIST_OPEN,
    PAIR_POS_PLACED,
    PAIR_POS_CLOSED,
    NUM_POS,
    ALL_SYM_OPEN,
    CLOSE_BUY,
    CLOSE_SELL,
    PAIR_LABEL,
    ACCOUNT_BALANCE
};
#endif // _CLIENT_API_FX_ACTION_H_
