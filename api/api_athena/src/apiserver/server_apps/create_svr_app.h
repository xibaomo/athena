/*
 * =====================================================================================
 *
 *       Filename:  create_svr_app.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/04/2018 17:06:15
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  fxua (), fxua@gmail.com
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _CREATE_SERVER_APP_H_
#define  _CREATE_SERVER_APP_H_
#include "server_apps/server_base_app/server_base_app.h"
#include "basics/types.h"

ServerBaseApp* create_server_app(AppType type, const String& configFile);
#endif   /* ----- #ifndef _CREATE_SERVER_APP_H_  ----- */
