#include "minbar_tracker.h"
#include "basics/utils.h"
#include <sstream>
#include <numeric>
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
    case FXAct::REQUEST_ACT_RTN:
        outmsg = procMsg_REQUEST_ACT_RTN(msg);
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

    auto tms = splitString(pack.str_vec[0],";");

    for (int i = 0; i < nbars; i++) {
        String ds,ts;
        if (!tms[i].empty()) {
            auto tmp = splitString(tms[i]," ");
            ds = tmp[0];
            ts=tmp[1];
        }
        MinBar mb{ds,ts,pm[0],pm[1],pm[2],pm[3],pm[4]};
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
    MinBar mb{tmp[0],tmp[1],round5pts(v[0]),
              round5pts(v[1]),
              round5pts(v[2]),
              round5pts(v[3]),v[4]};
    m_allMinBars.push_back(mb);
    m_predictor->appendMinbar(mb);

    stringstream iss;
    iss << mb.date << " "
        << mb.time << " "
        << mb.open << " "
        << mb.high << " "
        << mb.low  << " "
        << mb.close <<" "
        << mb.tickvol;
    Log(LOG_INFO) << "New minbar: " + iss.str() <<std::endl;

    Message outmsg;
    return outmsg;
}

Message
MinbarTracker::procMsg_REGISTER_POS(Message& msg) {
    m_numOpenPos++;

    SerializePack pack;
    unserialize(msg.getComment(),pack);

    mt5ulong ticket = pack.mt5ulong_vec[0];
    String tm = pack.str_vec[0];
    real64 ask = pack.real64_vec[0];
    real64 bid = pack.real64_vec[1];

    if (m_tk2pos.find(ticket) != m_tk2pos.end()) {
        Log(LOG_ERROR) << "Position already registered: " << ticket <<std::endl;
    }
    PosInfo pf;
    pf.open_time = tm;
    pf.open_ask = ask;
    pf.open_bid = bid;
    m_tk2pos[ticket] = pf;
    Log(LOG_INFO) << "Position registered:  " << tm << ", ticket: " << ticket <<std::endl;

    Message outmsg;
    return outmsg;
}

Message
MinbarTracker::procMsg_CLOSED_POS_INFO(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    mt5ulong tk = pack.mt5ulong_vec[0];
    String tm = pack.str_vec[0];
    real64 price = pack.real64_vec[0];
    real64 profit = pack.real64_vec[1];

    if (m_tk2pos.find(tk) == m_tk2pos.end()) {
        Log(LOG_ERROR) << "Ticket not registered: " + to_string(tk) <<std::endl;
    }

    auto& pos = m_tk2pos[tk];
    if (!pos.close_time.empty()) {
        Message outmsg;
        return outmsg;
    }
    m_numClosePos++;
    pos.close_time = tm;
    pos.profit = profit;
    pos.close_price = price;

    Log(LOG_INFO) << "Position closed. Start time: " << pos.open_time
                  << ", close time: " << pos.close_time << ", profit: " << profit << endl;

    Message outmsg;
    return outmsg;
}

Message
MinbarTracker::procMsg_REQUEST_ACT(Message& msg) {
    real64* pm = (real64*)msg.getData();
    FXAct act = m_predictor->predict(msg.getComment(), pm[0]);

    Message outmsg(1);
    outmsg.setAction(act);
    return outmsg;
}

Message
MinbarTracker::procMsg_REQUEST_ACT_RTN(Message& msg) {
    real64* pm = (real64*)msg.getData();
    FXAct act = m_predictor->predict(msg.getComment(), pm[0]);

    Message outmsg(sizeof(real64));
    outmsg.setAction(act);
    real64* pd = (real64*)outmsg.getData();
    pd[0] = m_predictor->getReturn();
    return outmsg;
}

void
MinbarTracker::finish() {
    Log(LOG_INFO) << "opened pos: " << m_numOpenPos << ", closed pos: " << m_numClosePos << endl;
    dumpPosInfo();
}

void
MinbarTracker::dumpPosInfo() {
    vector<String> start_times;
    vector<String> end_times;
    vector<real64> profits;
    vector<int> lifetimes;
    vector<unsigned long> tks;
    vector<real64> open_asks;
    vector<real64> open_bids;
    vector<real64> close_prices;
    for(auto iter : m_tk2pos) {
        tks.push_back(iter.first);
        start_times.push_back(iter.second.open_time);
        end_times.push_back(iter.second.close_time);
        profits.push_back(iter.second.profit);
        lifetimes.push_back(iter.second.lifetime());
        open_asks.push_back(iter.second.open_ask);
        open_bids.push_back(iter.second.open_bid);
        close_prices.push_back(iter.second.close_price);
    }

    // sort against starting time
    vector<int> dts;
    bt::ptime t0(bt::time_from_string(start_times[0]));
    for (auto s : start_times) {
        bt::ptime t(bt::time_from_string(s));
        auto dt = t - t0;
        dts.push_back(dt.total_seconds());
    }

    vector<int> ids(dts.size());
    std::iota(ids.begin(),ids.end(),0);
    std::sort(ids.begin(),ids.end(),[&](int i, int j) {
        return dts[i] < dts[j];
    });

    auto dts_aux = dts;
    auto tks_aux = tks;
    auto st_aux = start_times;
    auto et_aux = end_times;
    auto lf_aux = lifetimes;
    auto pf_aux = profits;
    auto oa_aux = open_asks;
    auto ob_aux = open_bids;
    auto cp_aux = close_prices;
    vector<int> guess(ids.size());
    for(size_t i=0; i < ids.size(); i++) {
        auto id = ids[i];
        dts_aux[i] = dts[id];
        tks_aux[i] = tks[id];
        st_aux[i]  = start_times[id];
        et_aux[i]  = end_times[id];
        lf_aux[i] = lifetimes[id];
        pf_aux[i] = profits[id];
        oa_aux[i] = open_asks[id];
        ob_aux[i] = open_bids[id];
        cp_aux[i] = close_prices[id];
        guess[i] = cp_aux[i]>oa_aux[i]? 1 : -1;
    }

    real64 profit = std::accumulate(pf_aux.begin(),pf_aux.end(),0.f);
    Log(LOG_INFO) << "Total profit: " << profit << endl;

    const String headers = "START_TIME,END_TIME,DURATION,PROFIT,OPEN_ASK,OPEN_BID,CLOSE_PRICE,LABEL";
    dumpVectors("pos_info.csv",headers, st_aux,et_aux,lf_aux,pf_aux,oa_aux,ob_aux,cp_aux,guess);
}
