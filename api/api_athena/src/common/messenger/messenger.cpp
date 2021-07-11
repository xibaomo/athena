/*
 * =====================================================================================
 *
 *       Filename:  messenger.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 04:45:57 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "messenger.h"
#include "sockutils.h"
#include "basics/utils.h"
#include "basics/log.h"
#include "conf/generalconf.h"

using namespace std;
using namespace athena;

Messenger::Messenger() : m_hostSock(-1), m_port(-1),
    m_bufferSize(-1), m_isListening(false)
{
    m_port = GeneralConfig::getInstance().getPort();

    m_hostSock = createTCPSocket();
    bindSocketToPort();
    enableListenSocket(m_hostSock);
    switchBlockingMode(m_hostSock, false);
    m_bufferSize = getBufferSize();
}

int
Messenger::createTCPSocket()
{
    boost::mutex::scoped_lock lock(m_mutex);
    int sock;

    if ( (sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP))<0 ) {
        Log(LOG_FATAL) << "Falied to create a tcp socket";
    }
    keepSockAlive(sock);

    return sock;
}

void
Messenger::keepSockAlive(int sock)
{
    int optval = 1;
    socklen_t optlen = sizeof(optval);
    if ( setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &optval, optlen) < 0 ) {
        Log(LOG_FATAL) << "Failed to set socket to keep alive";
    }
    return;
}

void
Messenger::bindSocketToPort(bool isGenPort)
{
    if ( !isGenPort ) {
        // use m_port
        Log(LOG_FATAL) << "Not supported";
    }

    //m_port = getpid();
    while ( 1 ) {
            //m_port = 8888;
        //m_port = m_port%(MAXPORTNUM-MINPORTNUM) + MINPORTNUM;
        memset((char*)&m_addr, 0, sizeof(m_addr));
        m_addr.sin_family = AF_INET;
        m_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        m_addr.sin_port = htons(m_port);
        if ( bind(m_hostSock, (struct sockaddr *)&m_addr, sizeof(m_addr))>=0 ) {
            Log(LOG_INFO) << "Bind successful. Port: " << m_port;
            break;
        }
        Log(LOG_VERBOSE) << "Tried port: " << m_port;
        //m_port++;
    }
}

void
Messenger::enableListenSocket(int sock, int backlog)
{
    if ( listen(sock, backlog)<0 ) {
        Log(LOG_FATAL) << "Failed to enable listening on socket";
    }
}

int
Messenger::getBufferSize()
{
    int sockbufsize = 0;
    socklen_t size = sizeof(int);
    int err = getsockopt(m_hostSock, SOL_SOCKET, SO_RCVBUF, (char*)&sockbufsize, &size);
    if ( err ) {
        Log(LOG_FATAL) << "Cannot obtain socket buffer size: " << sockbufsize;
    }

    return sockbufsize;
}

void
Messenger::drainSocket(int sock)
{
    char* buffer = new char[m_bufferSize];
    memset(buffer, 0, m_bufferSize);

    int valread = read(sock, buffer, m_bufferSize);
    if ( valread > 0 ) {
        char* buf = buffer;
        while ( buf ) {
            buf = readMsgFromSocket(sock, buf, valread, buffer);//return when a whole msg is read
        }
    }
    delete buffer;
    return;
}

char*
Messenger::readMsgFromSocket(int sock, char* buffer, int& valread, char* readBuffer)
{
    char* pb = buffer;
    size_t msgSize = *((size_t*)pb);
    Message msg;
    msg.setMsgSize(msgSize);

    if ( (int)msgSize <= valread ) {
        // valread contains the entire msg
        memmove(msg.getHead(), pb, msgSize);
        sendToSelf(msg);
        valread-=msgSize;
        if ( valread == 0 )
            return nullptr;
    } else {
        // read socket until a complete msg is read
        char* dest = (char*)msg.getHead();
        memmove(dest, pb, valread);
        dest+=valread;
        size_t sumBytes = valread;
        while ( 1 ) {
            memset(readBuffer, 0, m_bufferSize);
            valread = read(sock, readBuffer, m_bufferSize);
            if ( valread<=0 ) {
                sleep(ONE_MS);
                continue;
            }
            sumBytes += (size_t)valread;
            if ( sumBytes <= msgSize ) {
                memmove(dest, readBuffer, valread);
                dest+=valread;

                if ( sumBytes == msgSize ) {
                    sendToSelf(msg);
                    return nullptr;
                }
            } else {
                Log(LOG_FATAL) << "One connection allows only one msg";
            }
        }
    }

    return nullptr;
}

void
Messenger::acceptReadConn()
{
    SockAddr clntAddr;
    socklen_t addrlen = sizeof(clntAddr);
    while ( m_isListening ) {
        bool rt = checkSockReadable(m_hostSock, ONE_HUNDRED_MS); //100ms timeout
        if ( !rt ) {
            sleepMilliSec(ONE_HUNDRED_MS);
            continue;
        }

        // a new connection comes in
        int clntsock;
        if ( (clntsock = accept(m_hostSock, (struct sockaddr*)&clntAddr,
                        (socklen_t*)&addrlen))<0) {
            Log(LOG_FATAL) << "Failed to ccept connection";
        }
        drainSocket(clntsock);
        hangup(clntsock);
        close(clntsock);

        break;
    }
}

void
Messenger::hangup(int sock)
{
    send(sock, HANGUP.c_str(), HANGUP.size(), 0);
}

void
Messenger::sockSend(Message& msg, int sock)
{
    char* pc = (char*) msg.getHead();
    int remain = (int)msg.getMsgSize();
    while ( remain > 0 ) {
        // keep sending until all bytes are sent
        if ( !checkSockWriteable(sock, ONE_HUNDRED_MS) ) {
            continue;
        }

        auto ms = send(sock, pc, remain, 0);
        remain-=(int)ms;
        pc+=ms;
    }

    waitConfirm(sock);
}

void
Messenger::sendAMsgToAddr(Message& msg, SockAddr& addr)
{
    int sock = createTCPSocket();
    connectSockAddr(sock, &addr);
    sockSend(msg, sock);
    close(sock);
    Log(LOG_DEBUG) << "msg sent. socket closed";
}

void
Messenger::sendAMsgToHostPort(Message& msg, const String& hostPort)
{
    SockAddr addr;
    vector<String> s = splitString(hostPort, ":");
    createSockAddr(&addr, s[0], stoi(s[1]));
    sendAMsgToAddr(msg, addr);
}

int
Messenger::listenOnce(Message& msg)
{
    bool rt = checkSockReadable(m_hostSock, 1);
    if ( rt ) {
        int clntsock;
        if ( (clntsock = accept(m_hostSock, NULL,
                        NULL))<0) {
            Log(LOG_FATAL) << "Failed to accept connection";
        }
        drainSocket(clntsock);
        hangup(clntsock);
        close(clntsock);

        msg = std::move(m_msgBox.front());
        m_msgBox.pop();
        MsgAct action = (MsgAct)msg.getAction();
        if ( action == MsgAct::NORMAL_EXIT )
            return -1;

        return 1;
    }

    Log(LOG_DEBUG) << "Listened once";
    return 0;
}

void
Messenger::waitConfirm(int sock)
{
    const int LEN = HANGUP.size();
    char buffer[LEN];
    while ( 1 ) {
        int valread = read(sock, buffer, LEN);
        if ( valread>0 ) {
            String msg(buffer);
            if ( msg == HANGUP )
                break;
        }
        sleepMilliSec(ONE_MS);
    }
}
