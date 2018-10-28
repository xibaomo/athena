/*
 * =====================================================================================
 *
 *       Filename:  sockutils.h
 *
 *    Description:  utils for sock comm
 *
 *        Version:  1.0
 *        Created:  10/27/2018 02:53:57 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MESSENGER_SOCK_UTILS_H_
#define  _MESSENGER_SOCK_UTILS_H_
#include <sys/socket.h>
#include <netinet/ip.h>

/*-----------------------------------------------------------------------------
 *  communication functions
 *-----------------------------------------------------------------------------*/
void connectSockAddr(int sock, struct sockaddr_in* servAddr);

bool checkSockWriteable(int sock, int timeout);

bool checkSockReadable(int sock, int timeout);

bool switchBlockingMode(int sock, bool isBlocking);

String getIPfromSockAddr(struct sockaddr_in* sockAddr);

void createSockAddr(sockaddr_in* sockAddr_out, const String& hostname, int port);

#endif   /* ----- #ifndef _MESSENGER_SOCK_UTILS_H_  ----- */
