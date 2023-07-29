#include "graphloop.h"
#include "pyhelper.hpp"
#include "pyrunner/pyrunner.h"
#include "basics/utils.h"

using namespace std;
using namespace athena;

GraphLoop::GraphLoop(const String& cf) : ServerBaseApp(cf) {
    String path = String(getenv("ATHENA_HOME")) + "/py_algos/graphloop";
    PyEnviron::getInstance().appendSysPath(path);
    m_mod = PyRunner::getInstance().importModule("graphloop_api");
    if(!m_mod)
        Log(LOG_FATAL) << "Failed to import module: graphloop_api" << endl;

    Log(LOG_INFO) << "Module loaded: graphloop_api" << endl;
}

Message
GraphLoop::processMsg(Message& msg) {
    Message outmsg;
    FXAct act = (FXAct)msg.getAction();
    switch(act) {
    case FXAct::GLP_ALL_SYMS:
        outmsg = procMsg_GLP_ALL_SYMS(msg);
        break;

    case FXAct::GLP_NEW_QUOTE:
        outmsg = procMsg_GLP_NEW_QUOTE(msg);
        break;
    default:
        Log(LOG_FATAL) << "Action not recognized: " + to_string((int)act) << std::endl;
        break;
    }

    return outmsg;

}

Message
GraphLoop::procMsg_GLP_ALL_SYMS(Message& msg) {
    PyObject* init_func = PyObject_GetAttrString(m_mod,"init");
    if(!init_func)
        Log(LOG_FATAL) << "Failed to find py function: init" <<std::endl;

    PyObject* cf = Py_BuildValue("s",m_configFile.c_str());
    PyObject* args = Py_BuildValue("(O)",cf);
    PyObject* res = PyObject_CallObject(init_func,args);

    if(!res || !PyList_Check(res))
        Log(LOG_FATAL) << "Error when calling init()" << endl;

    SerializePack pack;

    Py_ssize_t listSize = PyList_Size(res);

    // Extract individual string elements from the list
    for (Py_ssize_t i = 0; i < listSize; ++i) {
        PyObject* pItem = PyList_GetItem(res, i);
        if (PyUnicode_Check(pItem)) {
            const char* strValue = PyUnicode_AsUTF8(pItem);
            pack.str_vec.push_back(String(strValue));
        }
        Py_DECREF(pItem);
    }

    Py_DECREF(res);
    Py_DECREF(args);
    Py_DECREF(cf);
    Py_DECREF(init_func);

    String cmt = serialize(pack);

    Message outmsg(FXAct::GLP_ALL_SYMS,cmt);
    return outmsg;
}

Message
GraphLoop::procMsg_GLP_NEW_QUOTE(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    PyObject* func = PyObject_GetAttrString(m_mod,"process");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: process" <<std::endl;

    npy_intp dims[2] = {1,pack.real64_vec.size()};
    PyObject* pArray = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    PyArrayObject* arr = (PyArrayObject*) pArray;
    real64* data = (real64*)arr->data;
    for (size_t i =0; i < dims[1]; i++) {
        data[i] = pack.real64_vec[i];
    }
    PyObject* arg1 = Py_BuildValue("s",pack.str_vec[0]);
    PyObject* args = Py_BuildValue("(OO)",arg1,arr);

    PyObject* pReturnTuple = PyObject_CallObject(func,args);

// Extract string list
    PyObject* pStringList = PyTuple_GetItem(pReturnTuple, 0);

// Extract int list
    PyObject* pIntList = PyTuple_GetItem(pReturnTuple, 1);
    // Convert string list
    std::vector<std::string> stringVec;
    for(int i=0; i<PyList_Size(pStringList); i++) {
        PyObject* pItem = PyList_GetItem(pStringList, i);
        stringVec.push_back(PyUnicode_AsUTF8(pItem));
    }

// Convert int list
    std::vector<int> intVec;
    for(int i=0; i<PyList_Size(pIntList); i++) {
        PyObject* pItem = PyList_GetItem(pIntList, i);
        intVec.push_back(PyLong_AsLong(pItem));
    }

// Clean up
    Py_DECREF(pArray);
    Py_DECREF(arg1);
    Py_DECREF(args);
    Py_DECREF(func);
    Py_DECREF(pStringList);
    Py_DECREF(pIntList);
    Py_DECREF(pReturnTuple);

    SerializePack outpack;
    outpack.int32_vec = std::move(intVec);
    outpack.str_vec = std::move(stringVec);

    String cmt = serialize(outpack);

    Message outmsg(FXAct::GLP_NEW_QUOTE,cmt);
    return outmsg;
}
