/*
 * =====================================================================================
 *
 *       Filename:  sockutils.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 02:55:44 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#include "basics/utils.h"
#include <poll.h>
#include <unistd.h>
#include <stdexcept>
#include <fcntl.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>
using namespace std;
using namespace athena;
void
connectSockAddr(int sock, struct sockaddr_in* servAddr)
{
    int ret(-1);
    while ( ret < 0 ) {
        ret = connect(sock, (struct sockaddr*)servAddr, sizeof(*servAddr));
        sleepMilliSec(ONE_MS);
    }
}

bool
checkSockWriteable(int sock,int timeout)
{
    pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLOUT;
    auto err = poll(&pfd,1,timeout);
    if (err < 0) {
        String msg = "[" + getHostName() + "-" + to_string(getpid()) + "] " +
                        "poll of socket fails\n";
        throw runtime_error(msg.c_str());
        return false;
    } else {
        if (pfd.revents & POLLOUT)
            return true;
    }

    return false;

}

bool
checkSockReadable(int sock,int timeout)
{
    pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;
    if (poll(&pfd,1,timeout) <0) { // 0 means timeout
        String msg = "[" + getHostName() + "-" + to_string(getpid()) + "] " +
                        "poll of socket fails\n";
        throw runtime_error(msg.c_str());
        return false;
    } else {
        if (pfd.revents & POLLIN)
            return true;
    }

    return false;
}

bool
switchBlockingMode(int sock, bool isBlocking)
{
    bool ret = true;
    const int flags = fcntl(sock, F_GETFL,0);
    if ((flags & O_NONBLOCK) && !isBlocking)
        // already in non-blocking mode
        return ret;

    if (!(flags & O_NONBLOCK) && isBlocking)
        // already in blocking mode
        return ret;

    ret = 0 == fcntl(sock,F_SETFL,
                     isBlocking? (flags^O_NONBLOCK):
                     (flags|O_NONBLOCK));
    return ret;

}

String
getIPfromSockAddr(struct sockaddr_in* addr)
{
    String ipp;
    char ipstr[INET6_ADDRSTRLEN];
    if (addr->sin_family == AF_INET) {
        int port = ntohs(addr->sin_port);
        inet_ntop(AF_INET, &addr->sin_addr,ipstr,sizeof(ipstr));
        ipp = String(ipstr) + ":" + to_string(port);
    } else {
        throw runtime_error("non-IPv4 address is not supported");
    }

    return ipp;
}

void
createSockAddr(sockaddr_in* outaddr, const String& hostname, int port)
{
    struct hostent* hp;
    hp = gethostbyname(hostname.c_str());
    if (!hp) {
        String msg = "Cannot obtain address of " + hostname;
        throw runtime_error(msg.c_str());
    } else {
        memset((char*)outaddr,0,sizeof(sockaddr_in));
        outaddr->sin_family = AF_INET;
        outaddr->sin_port = htons(port);
        memcpy((void*)&outaddr->sin_addr,hp->h_addr_list[0],hp->h_length);
    }
    return;
}
