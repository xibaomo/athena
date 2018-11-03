//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
#ifndef _API_SERVER_BASE_APP_H_
#define _API_SERVER_BASE_APP_H_

#include "app_base/app_base.h"
#include "basics/types.h"
#include "messenger/messenger.h"

class ServerBaseApp : public App {
protected:
    ServerBaseApp(const String& clientHostPort);

    String m_clientHostPort;
public:
    virtual ~ServerBaseApp() {;}

    virtual void prepare() = 0;

    void execute();

    virtual void finish() {;}

    virtual void processMsg(Message& msg) = 0;

    template <typename T>
    void sendBackResult(MsgAction action, T* p, const int len);
};

template <typename T>
void
ServerBaseApp::sendBackResult(MsgAction action, T* p, const int len)
{
    int databytes = sizeof(Real)*len;
    Message msg(databytes);
    msg.setAction(action);
    Real* pm = (Real*)msg.getData();
    for ( int i = 0; i < len; i++ ) {
        pm[i] = p[i];
    }

    m_msger->sendAMsgToHostPort(msg, m_clientHostPort);
}

#endif
