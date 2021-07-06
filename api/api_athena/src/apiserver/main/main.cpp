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
#include "conf/generalconf.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{

    PyEnviron::getInstance();
    Log(LOG_INFO) << "Athena api-server starts";

    // argv[1] is config file
    if (argc < 2)
        Log(LOG_FATAL) << "Usage: api_server <yaml_file>";

    GeneralConfig* cfg = &GeneralConfig::getInstance();
    cfg->loadConfig(String(argv[1]));

    Log::setLogLevel(cfg->getLogLevel());
    AppType atp = cfg->getAppType();

    ServerBaseApp* app = create_server_app(atp, String(argv[1]));

    app->prepare();

    app->execute();

    app->finish();

    Log(LOG_INFO) << "Athena api-server exits normally.";

    return 0;
}
