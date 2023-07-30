#pragma once

#include "server_apps/server_base_app/server_base_app.h"
#include <Python.h>
class GraphLoop : public ServerBaseApp {
private:
    PyObject* m_mod;
    GraphLoop(const String& cf);
public:

    static GraphLoop& getInstance(const String& cf="") {
        static GraphLoop _ins(cf);
        return _ins;
    }

    Message processMsg(Message& msg);

    Message procMsg_GLP_ALL_SYMS(Message& msg);
    Message procMsg_GLP_NEW_QUOTE(Message& msg);
};
