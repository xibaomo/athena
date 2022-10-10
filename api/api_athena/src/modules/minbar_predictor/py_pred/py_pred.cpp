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
MinbarPyPredictor::setPredictorFile(const String& path,const String& pf)
{
    Log(LOG_INFO) << "Predictor folder: " + path <<std::endl;
    PyEnviron::getInstance().appendSysPath(path);

    String modName = getFileStem(pf);

    m_mod = PyRunner::getInstance().importModule(modName);
}

void
MinbarPyPredictor::loadMinbarsToPredictor()
{
    PyObject* init_func = PyObject_GetAttrString(m_mod,"init");
    if(!init_func)
        Log(LOG_FATAL) << "Failed to find py function: init" <<std::endl;

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
        PyList_SetItem(date_list,i,Py_BuildValue("s",(*m_allMinBars)[i].date.c_str()));
        PyList_SetItem(time_list,i,Py_BuildValue("s",(*m_allMinBars)[i].time.c_str()));
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
MinbarPyPredictor::appendMinbar(const MinBar& mb)
{
    PyObject* func = PyObject_GetAttrString(m_mod,"appendMinbar");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: appendMinbar" <<std::endl;

    PyObject* date = Py_BuildValue("s",mb.date.c_str());
    PyObject* time = Py_BuildValue("s",mb.time.c_str());
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

int
MinbarPyPredictor::predict(const String& time_str, real64 new_open) {
    Log(LOG_VERBOSE) << "Server time: " << time_str << ", price: " << new_open << ", predicting ..." << endl;
    PyObject* func = PyObject_GetAttrString(m_mod,"predict");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: predict" <<std::endl;

    PyObject* np = Py_BuildValue("d",new_open);
    PyObject* ts = Py_BuildValue("s",time_str.c_str());
    PyObject* args = Py_BuildValue("(OO)",ts,np);

    PyObject* res = PyObject_CallObject(func,args);

    int  r =  (int)PyLong_AsLong(res);

    Log(LOG_VERBOSE) << "Action returned: " << r << ", Server time: " << time_str << endl;

//    FXAct act = FXAct::NOACTION;
//    switch(r) {
//    case 1:
//        Log(LOG_INFO) << "Decision: buy" << endl;
//        act = FXAct::PLACE_BUY;
//        break;
//    case 2:
//        Log(LOG_INFO) << "Decision: sell" << endl;
//        act = FXAct::PLACE_SELL;
//        break;
//    }

    Py_DECREF(func);
    Py_DECREF(np);
    Py_DECREF(ts);
    Py_DECREF(args);
    Py_DECREF(res);

    return r;
}

void
MinbarPyPredictor::finish() {
    PyObject* func = PyObject_GetAttrString(m_mod,"finalize");
    PyObject* args = 0;
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: finalize" <<std::endl;
    PyObject_CallObject(func,args);

    Py_DECREF(func);
}
