#pragma once

#include "server_apps/server_base_app/server_base_app.h"
#include <Python.h>
class GraphLoop : public ServerBaseApp {
private:
    PyObject* m_mod;
    GraphLoop(const String& cf);
public:

    static GraphLoop& getInstance(const String& cf="") {
        GraphLoop _ins(cf);
        return _ins;
    }

    Message processMsg(Message& msg);
};
