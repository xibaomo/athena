/*
 * =====================================================================================
 *
 *       Filename:  create_svr_app.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/04/2018 17:10:00
 *
 *         Author:  fxua (), fxua@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#include "create_svr_app.h"
#include "server_apps/server_predictor/server_predictor.h"
#include "server_apps/fx_tick_classifier/fx_tick_classifier.h"
#include "server_apps/fx_minbar_classifier/fx_minbar_classifier.h"
#include "server_apps/minbar_tracker/minbar_tracker.h"
#include "server_apps/minbar_pair_trader/minbar_pair_trader.h"
#include "server_apps/pair_selector/pair_selector.h"
#include "server_apps/robust_pair_trader/robust_pair_trader.h"
#include "server_apps/multinode_trader/multinode_trader.h"
#include "server_apps/pair_labeler/pair_labeler.h"

ServerBaseApp*
create_server_app(AppType type, const String& configFile)
{
    ServerBaseApp* app(nullptr);
    switch(type) {
        case AppType::APP_PREDICTOR:
            app = &ServerPredictor::getInstance(configFile);
            break;
        case AppType::APP_TICKCLASSIFIER:
            app = &ForexTickClassifier::getInstance(configFile);
            break;
        case AppType::APP_MINBARCLASSIFIER:
            app = &ForexMinBarClassifier::getInstance(configFile);
            break;
        case AppType::APP_PAIR_SELECTOR:
            app = &PairSelector::getInstance(configFile);
            break;
        case AppType::APP_MINBAR_TRACKER:
            app = &MinBarTracker::getInstance(configFile);
            Log(LOG_INFO) << "Min bar tracker created";
            break;
        case AppType::APP_MINBAR_PAIRTRADER:
            app = &MinbarPairTrader::getInstance(configFile);
            break;
        case AppType::APP_ROBUST_PAIRTRADER:
            app = &RobustPairTrader::getInstance(configFile);
            break;
        case AppType::APP_MULTINODE_TRADER:
            app = &MultinodeTrader::getInstance(configFile);
            break;
        case AppType::APP_PAIR_LABELER:
            app = &PairLabeler::getInstance(configFile);
            break;
        default:
            break;
    }

    return app;
}

