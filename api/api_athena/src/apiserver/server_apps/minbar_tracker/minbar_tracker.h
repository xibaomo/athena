#pragma once
#include "server_apps/server_base_app/server_base_app.h"
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include "create_mbp.h"
#include "mbtconf.h"
#include <map>
#include "boost/date_time/posix_time/posix_time.hpp"
namespace bt = boost::posix_time;
enum PosType {
    BUY,
    SELL
};
struct PosInfo {
    String open_time;
    String close_time;
    PosType pos_type;
    double profit;
    int lifetime() { // in unit of hours
        if (open_time.empty() || close_time.empty()){
            return -1;
        }
        bt::ptime ts(bt::time_from_string(open_time));
        bt::ptime te(bt::time_from_string(close_time));
        auto dt = te - ts;
        tm dpt = to_tm(dt);
        return dpt.tm_hour;
    }
};


class MinbarTracker : public ServerBaseApp
{
protected:

    std::map<unsigned long,PosInfo> m_tk2pos;
    MinBarBasePredictor* m_predictor;
    MbtConfig* m_mbtCfg;

    std::vector<MinBar> m_allMinBars;

    MinbarTracker(const String& cf) : m_predictor(nullptr), m_mbtCfg(nullptr), ServerBaseApp(cf)
    {
        m_mbtCfg = new MbtConfig();
        m_mbtCfg->loadConfig(cf);
        m_predictor = createMBPredictor(m_mbtCfg->getPredictorType(),m_mbtCfg);
        m_predictor->loadAllMinBars(&m_allMinBars);
    }
public:
    virtual ~MinbarTracker()
    {
        if (m_mbtCfg)
            delete m_mbtCfg;
        if (m_predictor)
            delete m_predictor;
    }
    static MinbarTracker& getInstance(const String& cf)
    {
        static MinbarTracker _ins(cf);
        return _ins;
    }

    void finish();
    void dumpPosInfo();

    Message processMsg(Message& msg);

    Message procMsg_HISTORY_MINBAR(Message& msg);
//    Message procMsg_INIT_TIME(Message& msg);
    Message procMsg_NEW_MINBAR(Message& msg);
    Message procMsg_REGISTER_POS(Message& msg);
    Message procMsg_CLOSED_POS_INFO(Message& msg);
    Message procMsg_REQUEST_ACTION(Message& msg);
};

