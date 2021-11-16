#include "builtin_ml.h"
#include "boost/date_time/posix_time/posix_time.hpp"
#include "pyrunner/pyrunner.h"
using namespace boost::posix_time;
using namespace std;

BuiltinMLPredictor::BuiltinMLPredictor() : m_mod(nullptr){
    String modulePath = String(getenv("ATHENA_HOME")) + "/minbar_classifier";
    PyEnviron::getInstance().appendSysPath(modulePath);
    Log(LOG_INFO) << "Python module path appended: " + modulePath;
    String modName = "minbar_api";
    m_mod = PyRunner::getInstance().importModule(modName);
    if(!m_mod) {
        Log(LOG_FATAL) << "Cannot import module: minbar_api";
    }

    m_pyPredictor.setPredictorFile(modulePath,modName);

    Log(LOG_INFO) << "Built-in ML predictor created";
}

void
BuiltinMLPredictor::loadConfig() {
    PyObject* str = Py_BuildValue("s",m_predConfigFile.c_str());

    PyObject* func = PyObject_GetAttrString(m_mod,"loadConfig");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: loadConfig";

    PyObject* args = Py_BuildValue("(O)",str);
    PyObject_CallObject(func,args);

    Py_DECREF(str);
    Py_DECREF(func);
    Py_DECREF(args);
}

