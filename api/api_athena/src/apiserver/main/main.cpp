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
#include "server_apps/create_svr_app.h"
#include "basics/log.h"

int main(int argc, char** argv)
{
    Log::setLogLevel(LOG_INFO);

    AppType atp = AppType::APP_PREDICTOR;

    Log(LOG_INFO) << "Athena api-server starts";
    ServerBaseApp* app = create_server_app(atp, String(argv[1]));

    app->prepare();

    app->execute();

    app->finish();

    Log(LOG_INFO) << "Athena api-server exits normally.";

    return 0;
}
