#pragma once
#include "server_apps/server_base_app/server_base_app.h"
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include "create_mbp.h"
#include "mbtconf.h"

class Obsolete_minbar_tracker : public ServerBaseApp {
protected:

    MinBarBasePredictor* m_predictor;
    MbtConfig* m_mbtCfg;

    std::vector<MinBar> m_allMinBars;

    std::vector<ActionRecord> m_actions;

    std::vector<real32> m_revenue;
    std::vector<real32> m_loss;

    real32 m_lowestProfit = 1.e6;

    Obsolete_minbar_tracker(const String& cfgFile) : m_predictor(nullptr), m_mbtCfg(nullptr), ServerBaseApp(cfgFile){
        m_mbtCfg = &MbtConfig::getInstance();
        m_mbtCfg->loadConfig(cfgFile);
        m_predictor = createMBPredictor(m_mbtCfg->getMinBarPredictorType(), cfgFile);
        m_predictor->loadAllMinBars(&m_allMinBars);
    }
public:
    virtual ~Obsolete_minbar_tracker();
    static Obsolete_minbar_tracker& getInstance(const String& cf) {
        static Obsolete_minbar_tracker _ins(cf);
        return _ins;
    }

    void dumpActions();
    void prepare();

    void loadMinBarFromFile(const String& barFile);

    Message processMsg(Message& msg);

    Message procMsg_HISTORY_MINBAR(Message& msg);
    Message procMsg_INIT_TIME(Message& msg);
    Message procMsg_MINBAR(Message& msg);
};
#endif   /* ----- #ifndef _SERVER_MINBAR_TRACKER_H_  ----- */
