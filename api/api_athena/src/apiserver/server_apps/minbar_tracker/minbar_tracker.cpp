#include "minbar_tracker.h"
using namespace std;
using namespace athena;

Message
MinbarTracker::processMsg(Message& msg) {
    Message outmsg;

    FXAct act = (FXAct)msg.getAction();
    switch(act) {
    case FXAct::HISTORY_MINBAR:
        outmsg = procMsg_HISTORY_MINBAR(msg);
        break;
    default:
        Log(LOG_FATAL) << "Action not recognized: " + to_string((int)act);
        break;
    }
    return outmsg;
}

Message
MinbarTracker::procMsg_HISTORY_MINBAR(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    int histLen = pack.int32_vec[0];
    if (histLen == 0) {
        Log(LOG_INFO) << "No min bars from mt5";
        Message out;
        return out;
    }

    int bar_size = pack.int32_vec[1];

    real32* pm = &pack.real32_vec[0];
    int nbars = msg.getDataBytes()/sizeof(real32)/bar_size;
    if (nbars != histLen) {
        Log(LOG_FATAL) << "No. of min bars inconsistent";
    }

    for (int i = 0; i < nbars; i++) {
        MinBar mb{"unknown","unknown",pm[0],pm[1],pm[2],pm[3],pm[4]};
        m_allMinBars.emplace_back(mb);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "Min bars from MT5 loaded: " + to_string(nbars);
    Log(LOG_INFO) << "Total history min bars: " + to_string(m_allMinBars.size());

    //display last 5 min bars
    int nb = 5;
    Log(LOG_INFO) << "Oldest 5 min bars";
    for (int i = m_allMinBars.size() - nbars; i < m_allMinBars.size() -  nbars + nb; i++) {
        auto& mb = m_allMinBars[i];
        showMinBar(mb);
    }
    Log(LOG_INFO) << "Latest 5 min bars: ";
    for (int i=m_allMinBars.size()-nb; i < (int)m_allMinBars.size(); i++) {
        auto& mb = m_allMinBars[i];
        showMinBar(mb);
    }

    m_predictor->prepare();
}
