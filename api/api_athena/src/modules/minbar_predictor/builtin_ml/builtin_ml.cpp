#include "builtin_ml.h"
#include "boost/date_time/posix_time/posix_time.hpp"
using namespace boost::posix_time;
using namespace std;

BuiltinMLPredictor::BuiltinMLPredictor(MbtConfig* cfg){
    m_predConfigFile = cfg->getKeyValue<String>("MINBAR_TRACKER/BUILTIN_ML/CONFIG_FILE");
    m_mod = PyImport_ImportModule("minbar_api");
    if(!m_mod) {
        Log(LOG_FATAL) << "Cannot import module: minbar_api";
    }
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

