#include "minbar_tracker.h"
#include "fx_action/fx_action.h"
#include "basics/csv_parser.h"
#include <fstream>
#include <numeric>
using namespace std;
using namespace athena;

MinBarTracker::~MinBarTracker()
{
    dumpActions();
}

void
MinBarTracker::dumpActions()
{
    ofstream ofs("actions.csv");
    ofs << "time,action"<<endl;
    for (auto& a : m_actions) {
        ofs << a.timestr<<"," << a.action << endl;
    }
    ofs.close();
}

void
MinBarTracker::prepare()
{
    loadMinBarFromFile(m_mbtCfg->getHistoryMinBarFile());
}

Message
MinBarTracker::processMsg(Message& msg)
{
    Message msgnew;
    FXAct action = (FXAct)msg.getAction();
    switch(action) {
    case FXAct::MINBAR:
        msgnew = procMsg_MINBAR(msg);
        break;
    case FXAct::HISTORY_MINBAR:
        msgnew = procMsg_HISTORY_MINBAR(msg);
        break;
    case FXAct::INIT_TIME:
        msgnew = procMsg_INIT_TIME(msg);
        break;
    case FXAct::PROFIT:
        msgnew = procMsg_noreply(msg,[this](Message& m){
        real32* pm = (real32*)m.getData();
        Log(LOG_INFO) << "Total profit of current positions: " + to_string(pm[0]);
        if (pm[0] < m_lowestProfit) m_lowestProfit = pm[0];

        Log(LOG_INFO) << "Lowest profit: " + to_string(m_lowestProfit);
                                 });
        break;

    case FXAct::CLOSE_POS:
        msgnew = procMsg_noreply(msg,[this](Message& m) {
        real32* pm = (real32*)m.getData();
        if (pm[0] > 0) m_revenue.push_back(pm[0]);
        else m_loss.push_back(pm[0]);

        real32 rev = std::accumulate(m_revenue.begin(),m_revenue.end(),0.);
        real32 loss = std::accumulate(m_loss.begin(),m_loss.end(),0.);
        Log(LOG_INFO) << "Earn: " + to_string(rev);
        Log(LOG_INFO) << "Loss: " + to_string(loss);
        Log(LOG_INFO) << "Net profit: " + to_string(rev + loss);
                                 });
        break;
    default:
        break;
    }

    return msgnew;
}

Message
MinBarTracker::procMsg_MINBAR(Message& msg)
{
    String timestr = msg.getComment();
    Log(LOG_INFO) << "New min bar arrives: " + timestr + " + 00:01";

    real32* pm = (real32*)msg.getData();

    m_allMinBars.emplace_back(timestr,pm[0],pm[1],pm[2],pm[3],pm[4]);

    showMinBar(m_allMinBars.back());

    FXAct action = m_predictor->predict();

    Log(LOG_INFO) << "Prediction: " + to_string((int)action);
    Message out;
    out.setAction(action);

    m_actions.emplace_back(timestr,(int)action);
    return out;
}

Message
MinBarTracker::procMsg_HISTORY_MINBAR(Message& msg)
{
    Log(LOG_INFO) << "Loading min bars from MT5 ...";

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
        m_allMinBars.emplace_back("unknown_time",pm[0],pm[1],pm[2],pm[3],pm[4]);
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

    Message outmsg;
    return outmsg;
}


Message
MinBarTracker::procMsg_INIT_TIME(Message& msg)
{
    String initTime = msg.getComment();
    Log(LOG_INFO) << "MT5 latest bar: " + initTime;

    String latestBarTime =  m_allMinBars.back().time;
    auto diffTime = getTimeDiffInMin(initTime,latestBarTime);

    int histLen;
    if (diffTime>0) {
        histLen = diffTime;
    } else {
        histLen = 0;
    }

    Message out(sizeof(int),latestBarTime.size());
    out.setComment(latestBarTime);
    int* pm = (int*)out.getData();
    pm[0] =  histLen;
    out.setAction(FXAct::REQUEST_HISTORY_MINBAR);
    Log(LOG_INFO) << "Request client to send history min bars: " + to_string(histLen);

    return out;
}

void
MinBarTracker::loadMinBarFromFile(const String& barFile)
{
    io::CSVReader<NUM_MINBAR_FIELDS> in(barFile);
    in.read_header(io::ignore_extra_column,"TIME","OPEN","HIGH","LOW","CLOSE","TICKVOL");
    String time;
    real32 open,high,low,close;
    int32 tickvol;
    while(in.read_row(time,open,high,low,close,tickvol)) {
        m_allMinBars.emplace_back(time,open,high,low,close,tickvol);
    }

    m_allMinBars.pop_back(); // drop the last min bar, which is normally incomplete
    Log(LOG_INFO) << "History min bars loaded from file: " + to_string(m_allMinBars.size());

    Log(LOG_INFO) << "Latest min bar in history: " + m_allMinBars.back().time;
}
