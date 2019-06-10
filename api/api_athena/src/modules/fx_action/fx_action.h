#ifndef _CLIENT_API_FX_ACTION_H_
#define _CLIENT_API_FX_ACTION_H_
#include "messenger/msg.h"

enum class FXAction {
    HISTORY = MsgAction::NUM_ACTIONS,
    HISTORY_MINBAR,
    CHECKIN,
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
    CLOSE_ALL_POS
};
#endif // _CLIENT_API_FX_ACTION_H_
