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

    // argv[0] is program name
    // argv[1] is client hostname:port
    // argv[2] is app type
    String clientHostPort = String(argv[1]);
    AppType atp = (AppType)stoi(String(argv[2]));

    Log(LOG_INFO) << "Athena api-server starts";
    ServerBaseApp* app = create_server_app(atp, clientHostPort);

    app->prepare();

    app->execute();

    app->finish();

    Log(LOG_INFO) << "Athena api-server exits normally.";

    return 0;
}
