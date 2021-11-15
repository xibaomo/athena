#include "server_apps/create_svr_app.h"
#include "basics/log.h"
#include "conf/generalconf.h"
#include "pyrunner/pyrunner.h"
#include <iostream>
#include "pyhelper.hpp"
int main() {
    PyEnviron::getInstance().appendSysPath("/home/naopc/dev/athena/minbar_classifier");
    return 0;
}
