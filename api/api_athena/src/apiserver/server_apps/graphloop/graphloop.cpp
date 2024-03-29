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

    Log(LOG_INFO) << "Graph loop created" << endl;
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
    case FXAct::GLP_GET_LOOP:
        outmsg = procMsg_GLP_GET_LOOP(msg);
        break;
    case FXAct::GLP_LOOP_RTN:
        outmsg = procMsg_GLP_LOOP_RTN(msg);
        break;
    case FXAct::GLP_CLEAR_LOOP:
        outmsg = procMsg_noreply(msg,[this](Message& msg) {
            m_loop.clear();
        });
        break;
    case FXAct::GLP_PROFIT_SLOPE:
        outmsg = procMsg_GLP_PROFIT_SLOPE(msg);
        break;
    case FXAct::GLP_SYM_CLOSE:
        outmsg = procMsg_GLP_SYM_CLOSE(msg);
        break;
    case FXAct::GLP_FINISH:
        outmsg = procMsg_GLP_FINISH(msg);
        break;
    default:
        Log(LOG_FATAL) << "Action not recognized: " + to_string((int)act) << std::endl;
        break;
    }

    return outmsg;

}

Message
GraphLoop::procMsg_GLP_ALL_SYMS(Message& msg) {
    Log(LOG_INFO) << "Start to get all symbols ..." << endl;

    PyObject* init_func = PyObject_GetAttrString(m_mod,"init");
    if(!init_func)
        Log(LOG_FATAL) << "Failed to find py function: init" <<std::endl;

    PyObject* cf = Py_BuildValue("s",m_configFile.c_str());
    PyObject* args = Py_BuildValue("(O)",cf);
    PyObject* res = PyObject_CallObject(init_func,args);

    if(!res || !PyList_Check(res))
        Log(LOG_FATAL) << "Error when calling init()" << endl;

    Py_ssize_t listSize = PyList_Size(res);
    size_t offset = 7;
    Message outmsg(FXAct::GLP_ALL_SYMS,0,offset*listSize);

    char* p = (char*)outmsg.getChar();
    // Extract individual string elements from the list
    for (Py_ssize_t i = 0; i < listSize; ++i) {
        PyObject* pItem = PyList_GetItem(res, i);
        if (PyUnicode_Check(pItem)) {
            const char* strValue = PyUnicode_AsUTF8(pItem);
            strcpy(p,strValue);
            p[offset-1]='\0';
            p+=offset;
        }
        Py_DECREF(pItem);
    }

    Py_DECREF(res);
    Py_DECREF(args);
    Py_DECREF(cf);
    Py_DECREF(init_func);

    Log(LOG_INFO) << "All symbols returned to client" << endl;

    return outmsg;
}

Message
GraphLoop::procMsg_GLP_NEW_QUOTE(Message& msg) {
    SerializePack pack;
    unserialize(msg.getComment(),pack);

    PyObject* func = PyObject_GetAttrString(m_mod,"process_quote");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: process_quote" <<std::endl;

    PyObject* ask_list = PyList_New(pack.real64_vec.size());
    PyObject* bid_list = PyList_New(pack.real64_vec.size());
    for(size_t i=0; i < pack.real64_vec.size(); i++) {
        PyList_SetItem(ask_list,i,Py_BuildValue("d",pack.real64_vec[i]));
        PyList_SetItem(bid_list,i,Py_BuildValue("d",pack.real64_vec1[i]));
    }


    PyObject* arg1 = Py_BuildValue("s",pack.str_vec[0].c_str());
    PyObject* args = Py_BuildValue("(OOO)",arg1,ask_list,bid_list);

    PyObject* pReturnTuple = PyObject_CallObject(func,args);

    if(!pReturnTuple)
        Log(LOG_FATAL) << "Execution of py func fails" << std::endl;

// Extract string list
    PyObject* pStringList = PyTuple_GetItem(pReturnTuple, 0);

// Extract int list
    PyObject* pIntList = PyTuple_GetItem(pReturnTuple, 1);

    PyObject* pPrice_list = PyTuple_GetItem(pReturnTuple,2);
    PyObject* pLot_list = PyTuple_GetItem(pReturnTuple, 3);

    int nsyms = PyList_Size(pStringList);
    if (nsyms == 0) {
        Message outmsg(FXAct::GLP_NEW_QUOTE,sizeof(int),0);
        int* pv = (int*)outmsg.getData();
        pv[0] = 0;
        return outmsg;
    }
    size_t offset = 7;
    Message outmsg(FXAct::GLP_NEW_QUOTE,sizeof(real64)*nsyms*3, offset*nsyms);
    char* pc = (char*)outmsg.getChar();
    real64* pv = (real64*)outmsg.getData();

    // Convert string list
    for(int i=0; i<PyList_Size(pStringList); i++) {
        PyObject* pItem = PyList_GetItem(pStringList, i);
        const char* tmp = PyUnicode_AsUTF8(pItem);
        strcpy(pc,tmp);
        pc[offset-1] = '\0';
        pc+=offset;
    }

// Convert data lists
    size_t pt=0;
    for(int i=0; i<PyList_Size(pIntList); i++) {
        PyObject* p1 = PyList_GetItem(pPrice_list,i);
        pv[pt++] = PyFloat_AsDouble(p1);
        PyObject* p2 = PyList_GetItem(pLot_list,i);
        pv[pt++] = PyFloat_AsDouble(p2);
        PyObject* pItem = PyList_GetItem(pIntList, i);
        pv[pt++] = (real64)PyLong_AsLong(pItem);
//        Py_DECREF(p1);
//        Py_DECREF(p2);
//        Py_DECREF(pItem);
    }

// Clean up
    Py_DECREF(ask_list);
    Py_DECREF(bid_list);
    Py_DECREF(arg1);
    Py_DECREF(args);
    Py_DECREF(func);
    Py_DECREF(pStringList);
    Py_DECREF(pIntList);
    Py_DECREF(pReturnTuple);
    Py_DECREF(pPrice_list);
    Py_DECREF(pLot_list);

    return outmsg;
}

Message
GraphLoop::procMsg_GLP_GET_LOOP(Message& msg) {
    PyObject* func = PyObject_GetAttrString(m_mod,"get_loop");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: get_loop" <<std::endl;

    PyObject* res = PyObject_CallObject(func,NULL);

    Py_ssize_t listSize = PyList_Size(res);
    size_t offset = 4;
    Message outmsg(FXAct::GLP_GET_LOOP,0,offset*listSize);

    char* p = (char*)outmsg.getChar();
    // Extract individual string elements from the list
    m_loop.clear();
    for (Py_ssize_t i = 0; i < listSize; ++i) {
        PyObject* pItem = PyList_GetItem(res, i);
        if (PyUnicode_Check(pItem)) {
            const char* strValue = PyUnicode_AsUTF8(pItem);
            m_loop.push_back(String(strValue));
            strcpy(p,strValue);
            p[offset-1]='\0';
            p+=offset;
        }
        Py_DECREF(pItem);
    }
    Py_DECREF(res);
    Py_DECREF(func);

    return outmsg;
}

Message
GraphLoop::procMsg_GLP_LOOP_RTN(Message& msg) {
    PyObject* func = PyObject_GetAttrString(m_mod,"get_loop_rtn");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: get_loop_rtn" <<std::endl;

    PyObject* res = PyObject_CallObject(func,NULL);
    if(!res) {
        Log(LOG_FATAL) << "Failed to run py funtion: get_loop_rtn" << std::endl;
    }

    double tw = PyFloat_AsDouble(res);

    Message outmsg(FXAct::GLP_LOOP_RTN,sizeof(double),0);
    double* pv = (double*)outmsg.getData();
    pv[0] = tw;

    Py_DECREF(res);
    Py_DECREF(func);
    return outmsg;
}

Message
GraphLoop::procMsg_GLP_PROFIT_SLOPE(Message& msg) {
    PyObject* func = PyObject_GetAttrString(m_mod,"compute_slope");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: compute_slope" <<std::endl;

    int np = msg.getDataBytes()/sizeof(real64);
    real64* pv = (real64*)msg.getData();
    PyObject* lx = PyList_New(np);
    PyObject* ly = PyList_New(np);
    for(size_t i=0; i < np; i++) {
        PyList_SetItem(lx,i,Py_BuildValue("d",i*1.));
        PyList_SetItem(ly,i,Py_BuildValue("d",pv[i]));
    }

    PyObject* args = Py_BuildValue("(OO)",lx,ly);

    PyObject* res = PyObject_CallObject(func,args);
    if(!res) {
        Log(LOG_FATAL) << "Failed to run py funtion: compute_slope" << std::endl;
    }

    double slp = PyFloat_AsDouble(res);

    Message outmsg(sizeof(real64),0);
    outmsg.setAction(FXAct::GLP_PROFIT_SLOPE);
    real64* opv = (real64*)outmsg.getData();
    opv[0] = slp;

    Py_DECREF(lx);
    Py_DECREF(ly);
    Py_DECREF(args);
    Py_DECREF(res);
    Py_DECREF(func);

    return outmsg;

}

Message
GraphLoop::procMsg_GLP_SYM_CLOSE(Message& msg){
    PyObject* func = PyObject_GetAttrString(m_mod,"find_sym_toclose");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: find_sym_toclose" <<std::endl;
    SerializePack pack;
    unserialize(msg.getComment(),pack);
    for(auto& s : pack.str_vec){
        cout << s << ", ";
    }
    cout<<endl;
    PyObject* arg1 = PyList_New(pack.real64_vec.size());
    auto& v = pack.real64_vec;
    for(size_t i=0;i < v.size();i++){
        PyList_SetItem(arg1,i,Py_BuildValue("d",v[i]));
    }

    PyObject* arg2 = Py_BuildValue("i",pack.int32_vec[0]);
    PyObject* arg3 = Py_BuildValue("i",pack.int32_vec[1]);

    PyObject* args = Py_BuildValue("(OOO)",arg1,arg2,arg3);

    PyObject* res = PyObject_CallObject(func,args);
    if(!res) {
        Log(LOG_FATAL) << "Failed to run py funtion: find_sym_toclose" << std::endl;
    }

    int id = PyLong_AsLong(res);

    Py_DECREF(arg1);
    Py_DECREF(arg2);
    Py_DECREF(arg3);
    Py_DECREF(args);
    Py_DECREF(res);
    Py_DECREF(func);

    Message outmsg(FXAct::GLP_SYM_CLOSE,pack.str_vec[id]);

    return outmsg;

}

Message
GraphLoop::procMsg_GLP_FINISH(Message& msg) {
    PyObject* func = PyObject_GetAttrString(m_mod,"finish");
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: finish" <<std::endl;

    PyObject_CallObject(func,NULL);
    Py_DECREF(func);

    Message outmsg;
    return outmsg;
}

