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
        default:
            break;
    }

    return app;
}

