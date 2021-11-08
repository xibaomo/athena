#include "py_pred.h"
/*************************************************************************
The python script must provide following functions:
1. init(date,time,open,high,low,close,tickvol), which takes initial data
    each argument is a list. 'date' and 'time' are string.
    The others are double

2. appendMinbar(date,time,open,high,low,close,tickvol), which appends
   a new entry of minbar but returns nothing

3. predict(new_open), which returns decision
    0 - no action
    1 - buy
    2 - sell

**************************************************************************/
#include "pyrunner/pyrunner.h"
#include "basics/utils.h"
using namespace std;
using namespace athena;

MinbarPyPredictor::MinbarPyPredictor(): m_mod(nullptr)
{
}

void
MinbarPyPredictor::setPredictorFile(const String& pf)
{
    String path = getFileFolder(pf);
    Log(LOG_INFO) << "Predictor folder: " + path;
    PyEnviron::getInstance().appendSysPath(path);

    String modName = getFileStem(pf);
    Log(LOG_INFO) << "module name: " + modName;

    m_mod = PyImport_ImportModule(modName.c_str());
    if (!m_mod)
        Log(LOG_FATAL) << "Failed to import module: " + modName;
}

void
MinbarPyPredictor::loadMinbarsToPredictor()
{
    PyObject* init_func = PyObject_GetAttrString(m_mod,"init");
    if(!init_func)
        Log(LOG_FATAL) << "Failed to find py function: init";

    size_t len =  m_allMinBars->size();
    PyObject* date_list = PyList_New(len);
    PyObject* time_list = PyList_New(len);
    PyObject* op = PyList_New(len);
    PyObject* hp = PyList_New(len);
    PyObject* lp = PyList_New(len);
    PyObject* cp = PyList_New(len);
    PyObject* tk = PyList_New(len);
    for (size_t i = 0; i < len; i++)
    {
        PyList_SetItem(date_list,i,Py_BuildValue("s",(*m_allMinBars)[i].date));
        PyList_SetItem(time_list,i,Py_BuildValue("s",(*m_allMinBars)[i].time));
        PyList_SetItem(op,i,Py_BuildValue("d",(*m_allMinBars)[i].open));
        PyList_SetItem(hp,i,Py_BuildValue("d",(*m_allMinBars)[i].high));
        PyList_SetItem(lp,i,Py_BuildValue("d",(*m_allMinBars)[i].low));
        PyList_SetItem(cp,i,Py_BuildValue("d",(*m_allMinBars)[i].close));
        PyList_SetItem(tk,i,Py_BuildValue("d",(*m_allMinBars)[i].tickvol));
    }
    PyObject* args = Py_BuildValue("(OOOOOOO)",date_list,time_list,op,hp,lp,cp,tk);

    PyObject_CallObject(init_func,args);

    Py_DECREF(init_func);
    Py_DECREF(date_list);
    Py_DECREF(time_list);
    Py_DECREF(op);
    Py_DECREF(hp);
    Py_DECREF(lp);
    Py_DECREF(cp);
    Py_DECREF(tk);
    Py_DECREF(args);
}

void
MinbarPyPredictor::addMinbarToPredictor(const MinBar& mb)
{
    PyObject* func = PyObject_GetAttrString(m_mod,"appendMinbar");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: appendMinbar";

    PyObject* date = Py_BuildValue("s",mb.date);
    PyObject* time = Py_BuildValue("s",mb.time);
    PyObject* op = Py_BuildValue("d",mb.open);
    PyObject* hp = Py_BuildValue("d",mb.high);
    PyObject* lp = Py_BuildValue("d",mb.low);
    PyObject* cp = Py_BuildValue("d",mb.close);
    PyObject* tk = Py_BuildValue("d",mb.tickvol);

    PyObject* args = Py_BuildValue("(OOOOOOO)",date,time,op,hp,lp,cp,tk);

    PyObject_CallObject(func,args);

    Py_DECREF(func);
    Py_DECREF(date);
    Py_DECREF(time);
    Py_DECREF(op);
    Py_DECREF(hp);
    Py_DECREF(lp);
    Py_DECREF(cp);
    Py_DECREF(tk);
    Py_DECREF(args);
}

void
MinbarPyPredictor::prepare()
{
    loadMinbarsToPredictor();
}

FXAct
MinbarPyPredictor::predict(real64 new_open) {
    PyObject* func = PyObject_GetAttrString(m_mod,"predict");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: predict";

    PyObject* np = Py_BuildValue("d",new_open);
    PyObject* args = Py_BuildValue("(O)",np);

    PyObject* res = PyObject_CallObject(func,args);

    int  r =  (int)PyLong_AsLong(res);
    switch(r) {
    case 1:
        return FXAct::PLACE_BUY;
    case 2:
        return FXAct::PLACE_SELL;
    }
    return FXAct::NOACTION;
}
