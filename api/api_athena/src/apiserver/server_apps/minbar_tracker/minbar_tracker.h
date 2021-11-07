#pragma once
#include "server_apps/server_base_app/server_base_app.h"
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include "create_mbp.h"
#include "mbtconf.h"

class MinbarTracker : public ServerBaseApp
{
protected:

    MinBarBasePredictor* m_predictor;
    MbtConfig* m_mbtCfg;

    std::vector<MinBar> m_allMinBars;

    MinbarTracker(const String& cf) : m_predictor(nullptr), m_mbtCfg(nullptr), ServerBaseApp(cf)
    {
        m_mbtCfg = new MbtConfig();
        m_mbtCfg->loadConfig(cf);
        m_predictor = createMBPredictor(m_mbtCfg->getPredictorFile());
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

    Message processMsg(Message& msg);

//    Message procMsg_HISTORY_MINBAR(Message& msg);
//    Message procMsg_INIT_TIME(Message& msg);
//    Message procMsg_MINBAR(Message& msg);
};

