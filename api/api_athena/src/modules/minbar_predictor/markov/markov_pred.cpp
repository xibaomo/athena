#include "markov_pred.h"
#include "pyhelper.hpp"
#include "pyrunner/pyrunner.h"

using namespace std;

MarkovPredictor::MarkovPredictor() : m_mod(nullptr) {
    String modPath = String(getenv("ATHENA_HOME")) + "/markov";
    PyEnviron::getInstance().appendSysPath(modPath);
    Log(LOG_INFO) << "Python module path appended: " << modPath << endl;

    String modName = "markov_api";
    m_mod = PyRunner::getInstance().importModule(modName);
    if(!m_mod) {
        Log(LOG_FATAL) << "Cannot import module: " << modName <<std::endl;
    }

    m_pyPredictor.setPredictorFile(modPath, modName);

    Log(LOG_INFO) << "Built-in markov predictor created" << endl;
}

void
MarkovPredictor::loadConfig() {
    PyObject* str = Py_BuildValue("s",m_predConfigFile.c_str());

    PyObject* func = PyObject_GetAttrString(m_mod,"loadConfig");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: loadConfig" <<std::endl;

    PyObject* args = Py_BuildValue("(O)",str);
    PyObject_CallObject(func,args);

    Py_DECREF(str);
    Py_DECREF(func);
    Py_DECREF(args);
}

real64
MarkovPredictor::getReturn() {
    String fn = "getReturn";
    PyObject* func = PyObject_GetAttrString(m_mod,fn.c_str());
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: " << fn <<std::endl;

    PyObject*  res = PyObject_CallObject(func,NULL);

    real64 rtn = PyFloat_AsDouble(res);

    Py_DECREF(func);
    Py_DECREF(res);

    return rtn;

}
