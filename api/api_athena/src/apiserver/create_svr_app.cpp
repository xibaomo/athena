//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
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
