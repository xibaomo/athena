#include "server_apps/create_svr_app.h"
#include "basics/log.h"
#include "conf/generalconf.h"
#include "pyrunner/pyrunner.h"
#include <iostream>
#include "pyhelper.hpp"
int main() {

    String athenaHome = String(getenv("ATHENA_HOME"));
    String p = athenaHome + "/minbar_classifier";
    PyEnviron::getInstance().appendSysPath(p.c_str());
    return 0;
}
