#include "mb_pairtrader.h"
#include <gsl/gsl_statistics_double.h>
using namespace std;

Message
MinBarPairTrader::processMsg(Message& msg)
{
    Message outmsg;
    FXAction action = (FXAction) msg.getAction();
    switch(action) {
    case FXAction::CHECKIN:
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            Log(LOG_INFO) << "Client checked in";
        });
        break;
    case FXAction::ASK_PAIR:
        outmsg = procMsg_ASK_PAIR(msg);
        break;
    case FXAction::PAIR_HIST_X:
        Log(LOG_INFO) << "X min bars arrive";
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            loadHistoryFromMsg(msg,m_minbarX);
            Log(LOG_INFO) << "Min bar X loaded";
        });
        break;
    case FXAction::PAIR_HIST_Y: {
        Log(LOG_INFO) << "Y min bars arrive";
        outmsg = procMsg_PAIR_HIST_Y(msg);
    }
    break;
    case FXAction::PAIR_MIN_OPEN:
        outmsg = procMsg_PAIR_MIN_OPEN(msg);
        break;
    default:
        break;
    }

    return outmsg;
}

Message
MinBarPairTrader::procMsg_PAIR_HIST_Y(Message& msg)
{
    Log(LOG_INFO) << "Y min bars arrive";

    loadHistoryFromMsg(msg,m_minbarY);
    Log(LOG_INFO) << "Min bar Y loaded";


    if (m_minbarX.size() != m_minbarY.size()) {
        Log(LOG_FATAL) << "Inconsistent length of X & Y";
    }

    real64 corr = computePairCorr();
    Log(LOG_INFO) << "Correlation: " + to_string(corr);

    linearReg();

    Message outmsg(msg.getAction(),sizeof(real32),0);
    real32* pm = (real32*) outmsg.getData();
    pm[0] = m_linregParam.c1;

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

    Message outmsg(FXAction::ASK_PAIR,sizeof(int),st.size());
    outmsg.setComment(st);

    int* pm = (int*)outmsg.getData();
    pm[0] = m_cfg->getLRLen();

    return outmsg;
}

void
MinBarPairTrader::linearReg()
{
    int len = m_minbarX.size();
    real64* x = new real64[len];
    real64* y = new real64[len];
    real64* spread = new real64[len];
    for (int i=0; i< len; i++) {
        x[i] = m_minbarX[i].open;
        y[i] = m_minbarY[i].open;
    }

    m_linregParam = linreg(x,y,len);
    Log(LOG_INFO) << "Linear regression done: c0 = " + to_string(m_linregParam.c0)
                  + ", c1 = " + to_string(m_linregParam.c1)
                  + ", sum_sq =  " + to_string(m_linregParam.chisq);

    for (int i=0; i < len; i++) {
        spread[i] = y[i] - m_linregParam.c1*x[i];
    }
    m_spreadMean = gsl_stats_mean(spread,1,len);
    real64 var  = gsl_stats_sd_m(spread,1,len,m_spreadMean);
    m_spreadStd = sqrt(var);

    real64 minsp,maxsp;
    gsl_stats_minmax(&minsp,&maxsp,spread,1,len);
    Log(LOG_INFO) << "Spread mean: " + to_string(m_spreadMean) + ", std: " + to_string(m_spreadStd);
    Log(LOG_INFO) << "Max deviation/std: " + to_string((minsp-m_spreadMean)/m_spreadStd) + ", "
                        + to_string((maxsp-m_spreadMean)/m_spreadStd);

    delete[] x;
    delete[] y;
    delete[] spread;
}

Message
MinBarPairTrader::procMsg_PAIR_MIN_OPEN(Message& msg)
{
    real32* pm = (real32*)msg.getData();
    real64 x = pm[0];
    real64 y = pm[1];

    char* pc = (char*)msg.getChar() + sizeof(int)*2;
    int cb = msg.getCharBytes() - sizeof(int)*2;
    String timeStr = String(pc,cb);

    Log(LOG_INFO) << "Mt5 time: " + timeStr + ", X: " + to_string(x)
                  + ", Y: " + to_string(y);

    real64 spread = y - m_linregParam.c1*x;

    real64 thd = m_cfg->getThresholdStd();

    real64 fac = (spread - m_spreadMean)/m_spreadStd;
    Log(LOG_INFO) << "err/sigma: " + to_string(fac);

    Message outmsg;
    if ( spread- m_spreadMean > thd*m_spreadStd) {
        outmsg.setAction(FXAction::PLACE_SELL);
    } else if( spread - m_spreadMean < -thd*m_spreadStd) {
        outmsg.setAction(FXAction::PLACE_BUY);
    } else {
        outmsg.setAction(FXAction::NOACTION);
    }

    return outmsg;
}

real64
MinBarPairTrader::computePairCorr()
{
    int len = m_minbarX.size();
    real64* x = new real64[len];
    real64* y = new real64[len];
    for (int i=0; i<len; i++) {
        x[i] = m_minbarX[i].open;
        y[i] = m_minbarY[i].open;
    }
    real64 corr = gsl_stats_correlation(x,1,y,1,len);

    return corr;
}
