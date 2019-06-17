#include "pyrunner/pyrunner.h"
#include "basics/utils.h"
using namespace std;
using namespace athena;

PyRunner::PyRunner()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/pyapi";

    m_pyInst.appendSysPath(modulePath);
}

CPyObject
PyRunner::runAthenaPyFunc(const String& modName, const String& funcName, CPyObject& args)
{
    CPyObject mod = PyImport_ImportModule(modName.c_str());
    if (!mod)
        Log(LOG_FATAL) << "Failed to import module: " + modName;

    CPyObject func = PyObject_GetAttrString(mod,funcName.c_str());
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: " + funcName;


    CPyObject res = PyObject_CallObject(func,args.getObject());

    return res;
}
