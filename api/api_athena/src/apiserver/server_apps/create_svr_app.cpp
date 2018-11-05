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

ServerBaseApp*
create_server_app(AppType type, const String& hp)
{
    ServerBaseApp* app(nullptr);
    switch(type) {
        case AppType::APP_PREDICTOR:
            app = &ServerPredictor::getInstance(hp);
            break;
        default:
            break;
    }

    return app;
}

