#include "minbar_tracker.h"
#include "basics/utils.h"
#include <sstream>
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
    case FXAct::NEW_MINBAR:
        outmsg = procMsg_NEW_MINBAR(msg);
        break;
    case FXAct::REGISTER_POS:
        outmsg = procMsg_REGISTER_POS(msg);
        break;
    case FXAct::CLOSE_POS_INFO:
        outmsg = procMsg_CLOSED_POS_INFO(msg);
        break;
    case FXAct::REQUEST_ACT:
        outmsg = procMsg_REQUEST_ACT(msg);
        break;
    default:
        Log(LOG_FATAL) << "Action not recognized: " + to_string((int)act) <<std::endl;
        break;
    }
    return outmsg;
}

Message
MinbarTracker::procMsg_HISTORY_MINBAR(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    int nbars = pack.int32_vec[0];
    if (nbars == 0) {
        Log(LOG_INFO) << "No min bars from mt5" <<std::endl;
        Message out;
        return out;
    }

    int bar_size = pack.int32_vec[1];

    real64* pm = &pack.real64_vec[0];

    for (int i = 0; i < nbars; i++) {
        MinBar mb{"unknown","unknown",pm[0],pm[1],pm[2],pm[3],pm[4]};
        m_allMinBars.emplace_back(mb);
        pm+=bar_size;
    }

    Log(LOG_INFO) << "Min bars from MT5 loaded: " + to_string(nbars) <<std::endl;
    Log(LOG_INFO) << "Total history min bars: " + to_string(m_allMinBars.size()) <<std::endl;

    //display last 5 min bars
    int nb = 5;
    Log(LOG_INFO) << "Oldest 5 min bars" <<std::endl;
    for (int i = m_allMinBars.size() - nbars; i < m_allMinBars.size() -  nbars + nb; i++) {
        auto& mb = m_allMinBars[i];
        showMinBar(mb);
    }
    Log(LOG_INFO) << "Latest 5 min bars: " <<std::endl;
    for (int i=m_allMinBars.size()-nb; i < (int)m_allMinBars.size(); i++) {
        auto& mb = m_allMinBars[i];
        showMinBar(mb);
    }

    m_predictor->prepare();

    Message outmsg;
    return outmsg;
}

Message
MinbarTracker::procMsg_NEW_MINBAR(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);
    String time_str = pack.str_vec[0];
    auto tmp = splitString(time_str," ");
    auto& v = pack.real64_vec;
    MinBar mb{tmp[0],tmp[1],v[0],v[1],v[2],v[3],v[4]};
    m_allMinBars.push_back(mb);
    m_predictor->appendMinbar(mb);

    stringstream iss; iss << mb.date << " "
                      << mb.time << " "
                      << mb.open << " "
                      << mb.high << " "
                      << mb.low  << " "
                      << mb.close <<" "
                      << mb.tickvol;
    Log(LOG_INFO) << "New minbar: " + iss.str() <<std::endl;

    m_predictor->appendMinbar(mb);
    Message outmsg;
    return outmsg;
}

Message
MinbarTracker::procMsg_REGISTER_POS(Message& msg) {
    mt5ulong *pm = (mt5ulong*)msg.getData();
    String tm = msg.getComment();
    if (m_tk2pos.find(pm[0]) != m_tk2pos.end()) {
        Log(LOG_ERROR) << "Position already registered: " << pm[0] <<std::endl;
    }
    PosInfo pf;
    pf.open_time = tm;
    m_tk2pos[pm[0]] = pf;
    Log(LOG_INFO) << "Position registered:  " << tm << ", ticket: " << pm[0] <<std::endl;

    Message outmsg;
    return outmsg;
}

Message
MinbarTracker::procMsg_CLOSED_POS_INFO(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    mt5ulong tk = pack.mt5ulong_vec[0];
    String tm = pack.str_vec[0];
    real64 profit = pack.real64_vec[0];

    if (m_tk2pos.find(tk) == m_tk2pos.end()) {
        Log(LOG_ERROR) << "Ticket not registered: " + to_string(tk) <<std::endl;
    }
    m_tk2pos[tk].close_time = tm;
    m_tk2pos[tk].profit = profit;

    Log(LOG_INFO) << "Position closed: " << m_tk2pos[tk].close_time << ", profit: " << profit << endl;

    Message outmsg;
    return outmsg;
}

Message
MinbarTracker::procMsg_REQUEST_ACT(Message& msg){
    real64* pm = (real64*)msg.getData();
    FXAct act = m_predictor->predict(pm[0]);

    Message outmsg(1);
    outmsg.setAction(act);
    return outmsg;
}

void
MinbarTracker::finish() {
    dumpPosInfo();
}

void
MinbarTracker::dumpPosInfo(){
    vector<String> start_times;
    vector<String> end_times;
    vector<real64> profits;
    vector<int> lifetimes;
    vector<unsigned long> tks;
    for(auto iter : m_tk2pos){
        tks.push_back(iter.first);
        start_times.push_back(iter.second.open_time);
        end_times.push_back(iter.second.close_time);
        profits.push_back(iter.second.profit);
        lifetimes.push_back(iter.second.lifetime());
    }
    dumpVectors("pos_info.csv",tks,start_times,end_times,lifetimes,profits);
}
