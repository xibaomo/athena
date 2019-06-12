#include "mb_pairtrader.h"
#include <gsl/gsl_statistics_double.h>
#include <fstream>
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
            loadHistoryFromMsg(msg,m_minbarX,m_openX);
            Log(LOG_INFO) << "Min bar X loaded";
        });
        break;
    case FXAction::PAIR_HIST_Y: {
        outmsg = procMsg_PAIR_HIST_Y(msg);
    }
    break;
    case FXAction::PAIR_MIN_OPEN:
        outmsg = procMsg_PAIR_MIN_OPEN(msg);
        break;

    case FXAction::SYM_HIST_OPEN:
        outmsg = procMsg_SYM_HIST_OPEN(msg);
        break;
    default:
        break;
    }

    return outmsg;
}
Message
MinBarPairTrader::procMsg_SYM_HIST_OPEN(Message& msg)
{
    String sym = msg.getComment();
    Log(LOG_INFO) << "Received history: " + sym;

    if (m_sym2hist.find(sym) != m_sym2hist.end()) {
        Log(LOG_FATAL) << "Duplicated symbol received: " + sym;
    }

    int len = msg.getDataBytes()/sizeof(real32);
    real32* pm = (real32*)msg.getData();
    std::vector<real32> v(pm,pm+len);
    m_sym2hist[sym] = std::move(v);

    Message out;
    return out;
}
Message
MinBarPairTrader::procMsg_PAIR_HIST_Y(Message& msg)
{
    Log(LOG_INFO) << "Y min bars arrive";

    loadHistoryFromMsg(msg,m_minbarY,m_openY);
    Log(LOG_INFO) << "Min bar Y loaded";


    if (m_minbarX.size() != m_minbarY.size()) {
        Log(LOG_FATAL) << "Inconsistent length of X & Y";
    }

    real64 corr = computePairCorr(m_openX,m_openY);
    Log(LOG_INFO) << "Correlation: " + to_string(corr);

    linearReg();

    Message outmsg(msg.getAction(),sizeof(real32),0);
    real32* pm = (real32*) outmsg.getData();
    pm[0] = m_linregParam.c1;

    return outmsg;
}
void
MinBarPairTrader::loadHistoryFromMsg(Message& msg, std::vector<MinBar>& v, std::vector<real32>& openvec)
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
        openvec.push_back(v.back().open);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "History min bars loaded: " + to_string(v.size());

}

Message
MinBarPairTrader::procMsg_ASK_PAIR(Message& msg)
{
    selectTopCorr();

    String s1 = m_cfg->getPairSymX();
    String s2 = m_cfg->getPairSymY();
    String st = s1 + ":" + s2;

    Message outmsg(FXAction::ASK_PAIR,sizeof(int),st.size());
    outmsg.setComment(st);

    return outmsg;
}

void
MinBarPairTrader::linearReg()
{
    int len = m_openX.size()-1;
    real64* x = new real64[len];
    real64* y = new real64[len];
    real64* spread = new real64[len];
    for (int i=0; i< len; i++) {
        x[i] = m_openX[i];
        y[i] = m_openY[i];
    }

    m_linregParam = linreg(x,y,len);
    Log(LOG_INFO) << "Linear regression done: c0 = " + to_string(m_linregParam.c0)
                  + ", c1 = " + to_string(m_linregParam.c1)
                  + ", sum_sq =  " + to_string(m_linregParam.chisq);

    for (int i=0; i < len; i++) {
        spread[i] = y[i] - m_linregParam.c1*x[i];
    }
    m_spreadMean = gsl_stats_mean(spread,1,len);
    m_spreadStd  = gsl_stats_sd_m(spread,1,len,m_spreadMean);

    real64 minsp,maxsp;
    gsl_stats_minmax(&minsp,&maxsp,spread,1,len);
    Log(LOG_INFO) << "Spread mean: " + to_string(m_spreadMean) + ", std: " + to_string(m_spreadStd);
    Log(LOG_INFO) << "Max deviation/std: " + to_string((minsp-m_spreadMean)/m_spreadStd) + ", "
                  + to_string((maxsp-m_spreadMean)/m_spreadStd);

    //dump spread
    ofstream ofs("spread.csv");
    ofs<<"x,y,spread"<<endl;
    for (int i=0; i < len; i++) {
        ofs << m_openX[i]<<","<<m_openY[i]<<","<<spread[i] << endl;
    }
    ofs.close();

    Log(LOG_INFO) << "spread dumped to spread.csv";
    delete[] x;
    delete[] y;
    delete[] spread;
}

Message
MinBarPairTrader::procMsg_PAIR_MIN_OPEN(Message& msg)
{
    real32* pm = (real32*)msg.getData();
    m_openX.push_back(pm[0]);
    m_openY.push_back(pm[1]);
    real64 x = pm[0];
    real64 y = pm[1];

    char* pc = (char*)msg.getChar() + sizeof(int)*2;
    int cb = msg.getCharBytes() - sizeof(int)*2;
    String timeStr = String(pc,cb);

    Log(LOG_INFO) << "Mt5 time: " + timeStr + ", X: " + to_string(x)
                  + ", Y: " + to_string(y);

    real64 corr = computePairCorr(m_openX,m_openY);
    Log(LOG_INFO) << "Correlation so far: " + to_string(corr);

    linearReg();

    real64 spread = y - m_linregParam.c1*x;

    real64 thd = m_cfg->getThresholdStd();

    real64 fac = (spread - m_spreadMean)/m_spreadStd;
    Log(LOG_INFO) << "err/sigma: " + to_string(fac);

    Message outmsg;
    if ( spread- m_spreadMean > thd*m_spreadStd) {
        outmsg.setAction(FXAction::PLACE_SELL);
    } else if( spread - m_spreadMean < -thd*m_spreadStd) {
        outmsg.setAction(FXAction::PLACE_BUY);
    } else if ( fac * m_prevSpread < 0) {
        outmsg.setAction(FXAction::CLOSE_ALL_POS);
    } else {
        outmsg.setAction(FXAction::NOACTION);
    }
    m_prevSpread = fac;
    return outmsg;
}

real64
MinBarPairTrader::computePairCorr(std::vector<real32>& v1, std::vector<real32>& v2)
{
    int len = v1.size();
    real64* x = new real64[len];
    real64* y = new real64[len];
    for (int i=0; i<len; i++) {
        x[i] = v1[i];
        y[i] = v2[i];
    }
    real64 corr = gsl_stats_correlation(x,1,y,1,len);

    delete[] x;
    delete[] y;

    return corr;
}

void
MinBarPairTrader::selectTopCorr()
{
    vector<String> keys;
    for(const auto& kv : m_sym2hist) {
        keys.push_back(kv.first);
    }

    for (size_t i = 0; i < keys.size(); i++) {
        for(size_t j=i+1; j < keys.size(); j++) {
            auto corr = computePairCorr(m_sym2hist[keys[i]],m_sym2hist[keys[j]]);
            if (corr > m_cfg->getCorrBaseline()) {
                SymPair sp{keys[i],keys[j],corr};
                m_topCorrSyms.push_back(sp);
                Log(LOG_INFO) << "Top coor pair: " + keys[i] + "," + keys[j] + ": " +to_string(corr);
            }
        }
    }
}
