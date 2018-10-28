/*
 * =====================================================================================
 *
 *       Filename:  utils.h
 *
 *    Description:  common utilities
 *
 *        Version:  1.0
 *        Created:  10/27/2018 02:31:18 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BASIC_UTILS_H_
#define  _BASIC_UTILS_H_

#include <sys/socket.h>
#include <netinet/ip.h>
#include <vector>
#include "types.h"

/*-----------------------------------------------------------------------------
 *  Execute system call by popen and return result as a string
 *-----------------------------------------------------------------------------*/
String execSysCall(const String& cmd);

/*-----------------------------------------------------------------------------
 *  Sleep in units of ms
 *-----------------------------------------------------------------------------*/
void sleepMilliSec(int num_ms);

/*-----------------------------------------------------------------------------
 *  Get local host name
 *-----------------------------------------------------------------------------*/
String getHostName();

/*-----------------------------------------------------------------------------
 *  Split a string
 *-----------------------------------------------------------------------------*/
std::vector<String>
splitString(const String& str, char delimiter=':');

#endif   /* ----- #ifndef _BASIC_UTILS_H_  ----- */
