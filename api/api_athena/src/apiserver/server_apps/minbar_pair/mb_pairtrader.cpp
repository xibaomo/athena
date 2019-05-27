#include "mb_pairtrader.h"
using namespace std;

Message
MinBarPairTrader::processMsg(Message& msg)
{
    Message outmsg;
    FXAction action = (FXAction) msg.getAction();
    switch(action) {
    case FXAction::ASK_PAIR:
        outmsg = procMsg_ASK_PAIR(msg);
        break;
    case FXAction::PAIR_HIST_X:
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            loadHistoryFromMsg(msg,m_minbarX);
            Log(LOG_INFO) << "Min bar X loaded";
        });
        break;
    case FXAction::PAIR_HIST_Y:
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            loadHistoryFromMsg(msg,m_minbarY);
            Log(LOG_INFO) << "Min bar Y loaded";
        });

        if (m_minbarX.size() != m_minbarY.size()) {
            Log(LOG_FATAL) << "Inconsistent length of X & Y";
        }
        linearReg();
        break;
    default:
        break;
    }

    return outmsg;
}

void
MinBarPairTrader::loadHistoryFromMsg(Message& msg, std::vector<MinBar>& v)
{

    int* pc = (int*)msg.getChar();
    int histLen = pc[0];
    if (histLen == 0) {
        Log(LOG_INFO) << "No min bars from mt5";
        return;
    }

    int bar_size = NUM_MINBAR_FIELDS-1;

    if (pc[1] != bar_size) {
        Log(LOG_FATAL) << "Min bar size inconsistent. MT5: " +  to_string(pc[1])
                       + ", local: " + to_string(bar_size);
    }
    real32* pm = (real32*) msg.getData();
    int nbars = msg.getDataBytes()/sizeof(real32)/bar_size;
    if (nbars != histLen) {
        Log(LOG_FATAL) << "No. of min bars inconsistent";
    }

    for (int i = 0; i < nbars; i++) {
        v.emplace_back("unknown_time",pm[0],pm[1],pm[2],pm[3],pm[4]);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History min bars loaded: " + to_string(v.size());

}

Message
MinBarPairTrader::procMsg_ASK_PAIR(Message& msg)
{
    String s1 = m_cfg->getPairSymX();
    String s2 = m_cfg->getPairSymY();
    String st = s1 + ":" + s2;

    Message outmsg(FXAction::ASK_PAIR,0,st.size());
    outmsg.setComment(st);
    return outmsg;
}

void
MinBarPairTrader::linearReg()
{

}

