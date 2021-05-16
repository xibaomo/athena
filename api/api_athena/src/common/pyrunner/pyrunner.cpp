#include "pyrunner/pyrunner.h"
#include "basics/utils.h"
using namespace std;
using namespace athena;
static void import_numpy()
{
    import_array();
}
PyRunner::PyRunner()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/pyapi";

    PyEnviron::getInstance().appendSysPath(modulePath);
}

CPyObject
PyRunner::runAthenaPyFunc(const String& modName, const String& funcName, CPyObject& args)
{
    CPyObject mod = PyImport_ImportModule(modName.c_str());
    if (!mod)
        Log(LOG_FATAL) << "Failed to import module: " + modName;

    CPyObject func = PyObject_GetAttrString(mod.getObject(),funcName.c_str());
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: " + funcName;


    CPyObject res = PyObject_CallObject(func.getObject(),args.getObject());

    return res;
}
