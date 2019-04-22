#include "minbar_tracker.h"
#include "fx_action/fx_action.h"
#include "basics/csv_parser.h"
using namespace std;
using namespace athena;

void
MinBarTracker::prepare()
{
    loadMinBarFromFile(m_mbtCfg->getHistoryMinBarFile());
}

Message
MinBarTracker::processMsg(Message& msg)
{
    Message msgnew;
    FXAction action = (FXAction)msg.getAction();
    switch(action) {
    case FXAction::MINBAR:
        msgnew = procMsg_MINBAR(msg);
        break;
    case FXAction::CHECKIN:
        msgnew = procMsg_noreply(msg,[this](Message& m){
                                 Log(LOG_INFO) << "Client checked in";
                                 });
        break;
    case FXAction::HISTORY_MINBAR:
        msgnew = procMsg_HISTORY_MINBAR(msg);
        break;
    case FXAction::INIT_TIME:
        msgnew = procMsg_INIT_TIME(msg);
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

    FXAction action = m_predictor->predict();
    Message out;
    out.setAction(action);
    return out;
}

Message
MinBarTracker::procMsg_HISTORY_MINBAR(Message& msg)
{
    Log(LOG_INFO) << "Loading min bars from MT5 ...";

    int* pc = (int*)msg.getChar();
    int histLen = pc[0];
    if (histLen == 0) {
        Log(LOG_INFO) << "No min bars from mt5";
        Message out;
        return out;
    }

    if (pc[1] != NUM_MINBAR_FIELDS-1) {
        Log(LOG_FATAL) << "Min bar size inconsistent. MT5: " +  to_string(pc[1])
         + ", local: " + to_string(NUM_MINBAR_FIELDS-1);
    }
    real32* pm = (real32*) msg.getData();
    for (int i = 0; i < msg.getDataBytes()/sizeof(real32); i+=(NUM_MINBAR_FIELDS-1)) {
        m_allMinBars.emplace_back("unknown_time",pm[0],pm[1],pm[2],pm[3],pm[4]);
    }

    Log(LOG_INFO) << "Min bars from MT5 loaded.";
    Log(LOG_INFO) << "Total history min bars: " + to_string(m_allMinBars.size());

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
    out.setAction(FXAction::REQUEST_HISTORY_MINBAR);
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
    Log(LOG_INFO) << "History min bars loaded from file: " + to_string(m_allMinBars.size());

    Log(LOG_INFO) << "Latest min bar in history: " + m_allMinBars.back().time;
}
