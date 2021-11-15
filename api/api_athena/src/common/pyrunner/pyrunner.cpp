#include "pyrunner/pyrunner.h"
#include "basics/utils.h"
using namespace std;
using namespace athena;
static void import_numpy()
{
    //import_array();
}
PyRunner::PyRunner()
{
    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/pyapi";

    PyEnviron::getInstance().appendSysPath(modulePath);
    Log(LOG_INFO) << "Added to python path: " + modulePath;

//    modulePath = athenaHome + "/minbar_classifier";
//    PyEnviron::getInstance().appendSysPath(modulePath);
//    Log(LOG_INFO) << "Added to python path: " + modulePath;
}

PyObject*
PyRunner::runAthenaPyFunc(const String& modName, const String& funcName, PyObject* args)
{
    //PyObject* mod = PyImport_Import(PyUnicode_FromString(modName.c_str()));
    PyObject* mod = PyImport_ImportModule(modName.c_str());
    if (!mod)
        Log(LOG_FATAL) << "Failed to import module: " + modName;

    PyObject* func = PyObject_GetAttrString(mod,funcName.c_str());
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: " + funcName;


    PyObject* res = PyObject_CallObject(func,args);

    Py_DECREF(mod);
    Py_DECREF(func);

    return res;
}

PyObject*
PyRunner::importModule(const String& modName) {
    if (m_addModules.find(modName)!=m_addModules.end())
        return m_addModules[modName];

    PyObject* mod = PyImport_ImportModule(modName.c_str());
    if (!mod)
        Log(LOG_FATAL) << "Failed to import module: " + modName;

    m_addModules[modName] = mod;
    return mod;
}
