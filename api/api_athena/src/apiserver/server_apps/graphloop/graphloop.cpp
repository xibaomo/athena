#include "graphloop.h"
#include "pyhelper.hpp"
#include "pyrunner/pyrunner.h"
using namespace std;

GraphLoop::GraphLoop(const String& cf) : ServerBaseApp(cf){
    String path = String(getenv("ATHENA_HOME")) + "/py_algos/graphloop";
    PyEnviron::getInstance().appendSysPath(path);
    m_mod = PyRunner::getInstance().importModule("graphloop_api");
    if(!m_mod)
        Log(LOG_FATAL) << "Failed to import module: graphloop_api" << endl;

    Log(LOG_INFO) << "Module loaded: graphloop_api" << endl;
}
