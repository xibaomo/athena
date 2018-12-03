/*
 * =====================================================================================
 *
 *       Filename:  app.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/02/2018 02:54:09 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "app.h"
#include "bar_marker/bar_marker.h"
App* createApp(AppType atp,const String& cfg)
{
    App* app(nullptr);
    switch(atp) {
    case AppType::BAR_MARKER:
        app = &BarMarker::getInstance(cfg);
        break;
    default:
        break;
    }

    return app;
}
