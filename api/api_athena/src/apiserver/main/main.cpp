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
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
    Log::setLogLevel(LOG_INFO);

    AppType atp = AppType::APP_MINBARCLASSIFIER;

    Log(LOG_INFO) << "Athena api-server starts";

    // argv[1] is config file
    if (argc < 2)
        Log(LOG_FATAL) << "Usage: api_server yaml_file";

    ServerBaseApp* app = create_server_app(atp, String(argv[1]));

    app->prepare();

    app->execute();

    app->finish();

    Log(LOG_INFO) << "Athena api-server exits normally.";

    return 0;
}
