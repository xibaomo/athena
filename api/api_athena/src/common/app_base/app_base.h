#ifndef APP_BASE_H_INCLUDED
#define APP_BASE_H_INCLUDED
#include "messenger/messenger.h"
enum AppType {
    APP_PREDICTOR = 0,
    APP_TICKCLASSIFIER
};

class App{
protected:
    App() { m_msger = &Messenger::getInstance(); }
    Messenger* m_msger;
public:
    virtual ~App() {;}
};

#endif
