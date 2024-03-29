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
#include "pyrunner/pyrunner.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
    Log(LOG_INFO) << "Athena api-server starts" <<std::endl;
    PyRunner::getInstance().runAthenaPyFunc("pd_utils","hello",NULL);

    // argv[1] is config file
    if (argc < 2)
        Log(LOG_FATAL) << "Usage: "<< argv[0] << " <yaml_file>" <<std::endl;

    GeneralConfig* cfg = &GeneralConfig::getInstance();
    cfg->loadConfig(String(argv[1]));

    Log::setLogLevel(cfg->getLogLevel());
    AppType atp = cfg->getAppType();

    ServerBaseApp* app = create_server_app(atp, String(argv[1]));

    app->prepare();

    app->execute();

    app->finish();

    Log(LOG_INFO) << "Athena api-server exits normally." <<std::endl;

    return 0;
}
