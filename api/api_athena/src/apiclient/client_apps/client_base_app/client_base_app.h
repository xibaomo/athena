/*
 * =====================================================================================
 *
 *       Filename:  client_base_app.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/04/2018 18:08:01
 *
 *         Author:  fxua (), fxua@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */
#ifndef  _CLIENT_BASE_APP_H_
#define  _CLIENT_BASE_APP_H_

#include "app_base/app_base.h"
#include "messenger/messenger.h"
class ClientBaseApp : public App {
protected:
    ClientBaseApp();

    String m_serverHostPort;
public:
    virtual ~ClientBaseApp();

    virtual void prepare(const String& mf) = 0;

    virtual void execute() = 0;

    virtual void finish() = 0;

    /*
     * Wake up api server
     */
    void wakeupServer(AppType atp);

    /*
     * Send message to api server to ask it to exit
     */
    void shutdownServer();
};
#endif   /* ----- #ifndef _CLIENT_BASE_APP_H_  ----- */
