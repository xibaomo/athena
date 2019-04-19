#ifndef APP_BASE_H_INCLUDED
#define APP_BASE_H_INCLUDED
#include "messenger/messenger.h"
#include "basics/types.h"

class App{
protected:
    App() { m_msger = &Messenger::getInstance(); }
    Messenger* m_msger;
public:
    virtual ~App() {;}
};

#endif
