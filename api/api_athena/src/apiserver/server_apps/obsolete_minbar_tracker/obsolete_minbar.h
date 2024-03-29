/*
 * =====================================================================================
 *
 *       Filename:  minbar_tracker.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/19/2019 12:53:17 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _SERVER_OBSOLETE_MINBAR_TRACKER_H_
#define  _SERVER_OBSOLETE_MINBAR_TRACKER_H_

#include "server_apps/server_base_app/server_base_app.h"
#include "minbar_predictor/mb_base/mb_base_pred.h"
#if 0

struct ActionRecord {
    String timestr;
    int action;
    ActionRecord(String& ts, int ac) : timestr(ts),action(ac){;}
};

class Obsolete_minbar_tracker : public ServerBaseApp {
protected:

    MinBarBasePredictor* m_predictor;
    MbtConfig* m_mbtCfg;

    std::vector<MinBar> m_allMinBars;

    std::vector<ActionRecord> m_actions;

    std::vector<real64> m_revenue;
    std::vector<real64> m_loss;

    real64 m_lowestProfit = 1.e6;

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

#endif // 0

#endif   /* ----- #ifndef _SERVER_MINBAR_TRACKER_H_  ----- */
