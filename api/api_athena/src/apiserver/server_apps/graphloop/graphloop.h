#pragma once

#include "server_apps/server_base_app/server_base_app.h"
#include <Python.h>
#include <vector>
class GraphLoop : public ServerBaseApp {
private:
    PyObject* m_mod;
    std::vector<String> m_loop;
    GraphLoop(const String& cf);
public:

    static GraphLoop& getInstance(const String& cf="") {
        static GraphLoop _ins(cf);
        return _ins;
    }

    Message processMsg(Message& msg);

    Message procMsg_GLP_ALL_SYMS(Message& msg);
    Message procMsg_GLP_NEW_QUOTE(Message& msg);
    Message procMsg_GLP_GET_LOOP(Message& msg);
    Message procMsg_GLP_LOOP_RTN(Message& msg);
    Message procMsg_GLP_PROFIT_SLOPE(Message& msg);
    Message procMsg_GLP_CLEAR_LOOP(Message& msg);
    Message procMsg_GLP_FINISH(Message& msg);
};
