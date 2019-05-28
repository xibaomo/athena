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
    case FXAction::PAIR_MIN_OPEN:
        outmsg = procMsg_PAIR_MIN_OPEN(msg);
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
    int len = m_minbarX.size();
    real64* x = new real64[len];
    real64* y = new real64[len];
    for (int i=0; i< len; i++) {
        x[i] = m_minbarX[i].open;
        y[i] = m_minbarY[i].open;
    }

    m_linregParam = linreg(x,y,len);
    Log(LOG_INFO) << "Linear regression done: c0 = " + to_string(m_linregParam.c0)
                    + ", c1 = " + to_string(m_linregParam.c1)
                    + ", sum_sq =  " + to_string(m_linregParam.chisq);
}

Message
MinBarPairTrader::procMsg_PAIR_MIN_OPEN(Message& msg)
{
    real32* pm = (real32*)msg.getData();
    real64 x = pm[0];
    real64 y = pm[1];

    real64 yp,yp_err;
    linreg_est(m_linregParam,x,&yp,&yp_err);

    real64 thd = m_cfg->getThresholdStd();

    Message outmsg;
    if (y - yp > thd*yp_err) {
        outmsg.setAction(FXAction::PLACE_SELL);
    } else if( y - yp < -thd*yp_err) {
        outmsg.setAction(FXAction::PLACE_BUY);
    } else {
        outmsg.setAction(FXAction::NOACTION);
    }

    return outmsg;
}
