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

void
BuiltinMLPredictor::setHourTimeID() {
    const size_t LEN =10;
    size_t hours = m_hourTimeID.size();
    PyObject* lx = PyList_New(LEN);
    for(size_t i=hours-1; i > hours-LEN;i--) {
        PyList_SetItem(lx,i,Py_BuildValue("I",m_hourTimeID[i]));
    }

    PyObject* func = PyObject_GetAttrString(m_mod,"setHourTimeID");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: setHourTimeID";
    PyObject* args = Py_BuildValue("(O)",lx);
    PyObject_CallObject(func,args);

    Py_DECREF(lx);
    Py_DECREF(func);
    Py_DECREF(args);
}

void
BuiltinMLPredictor::pushHourID(const MinBar& mb) {
    size_t len = m_allMinBars->size();
    auto& latest_mb = (*m_allMinBars)[len-1];
    ptime t(time_from_string(mb.date + " " + mb.time));
    tm pt = to_tm(t);
    if (pt.tm_min < 2) {
        m_hourTimeID.push_back(len-1);
    }
}
