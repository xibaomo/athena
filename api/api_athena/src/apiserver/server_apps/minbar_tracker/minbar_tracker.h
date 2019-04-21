/*
 * =====================================================================================
 *
 *       Filename:  minbar_tracker.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/19/2019 12:53:17 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _SERVER_MINBAR_TRACKER_H_
#define  _SERVER_MINBAR_TRACKER_H_

#include "server_apps/server_base_app/server_base_app.h"
struct MinBar {
    String time;
    real32 open;
    real32 high;
    real32  low;
    real32 close;
    int32  tickvol;
};

class MinBarTracker : public ServerBaseApp {
protected:
    std::vector<MinBar> m_allMinBars;
    MinBarTracker(const String& cfgFile) : ServerBaseApp(cfgFile){;}
public:
    virtual ~MinBarTracker(){;}
    static MinBarTracker& getInstance(const String& cf) {
        static MinBarTracker _ins(cf);
        return _ins;
    }

    void prepare(){;}

    Message processMsg(Message& msg) {Message m; return m;}
};
#endif   /* ----- #ifndef _SERVER_MINBAR_TRACKER_H_  ----- */
