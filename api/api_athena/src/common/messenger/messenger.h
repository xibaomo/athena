/*
 * =====================================================================================
 *
 *       Filename:  messenger.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 04:37:37 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BASIC_MESSENGER_H_
#define  _BASIC_MESSENGER_H_

#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <vector>
#include "msg.h"
#include "basics/types.h"
#include "basics/mtqueue.h"
const Uint MAXPORTNUM = 8899; //65536;
const Uint MINPORTNUM = 8800; //1024;
const String HANGUP = "hangup";
typedef struct sockaddr_in SockAddr;

class Messenger {
protected:
    int m_hostSock; // bind to port, used for listen
                    // to connect, create a new socket in situ
    int m_port;

    SockAddr m_addr;

    int m_bufferSize;

    bool m_isListening;

    MtQueue<Message> m_msgBox;

    boost::mutex m_mutex;

    Messenger();

public:
    virtual ~Messenger() {;}

    static Messenger& getInstance() {
        static Messenger _instance;
        return _instance;
    }

    int getHostSocket() const { return m_hostSock; }
    int getPort() const { return m_port; }

    Message popMsgBox() {
        Message msg = std::move(m_msgBox.front());
        m_msgBox.pop();
        return msg;
    }

    void shutdownConnection(int sock) { shutdown(sock,SHUT_WR); }

    void writeSocket(int sock, Message& msg)
    {
        int res = send(sock, (char*)msg.getHead(), (int)msg.getMsgSize(),0);
        if (res < 0) {
            Log(LOG_FATAL) << "Send failed.";
            return;
        }
    }

    /*
     * Send msg to sock address
     */
    void sendAMsgToAddr(Message& msg, SockAddr& addr);

    /*
     * Send msg to hostname:port
     */
    void sendAMsgToHostPort(Message& msg, const String& hostnamePort);

    /*
     * push a msg to local msg box
     */
    void sendToSelf(Message& msg)
    {
        m_msgBox.push(std::move(msg));
    }

    /*
     * If isGenPort is true, find an available port
     * otherwise use m_port;
     */
    void bindSocketToPort(bool isGenPort = true);

    int createTCPSocket();
    void keepSockAlive(int sock);
    void enableListenSocket(int sock, int backlog = 16);

    int getBufferSize();

    /*
     *     Accept and read incoming connection
     */
    void acceptReadConn();

    /*
     * Keep reading a socket until the entire message is read
     */
    void drainSocket(int sock);

    /*
     * only read one msg from buffer
     */
    char* readMsgFromSocket(int sock, char* buffer, int& valread, char* readBuffer);

    /*
     * send msg via socket
     */
    void sockSend(Message& msg, int sock);

    /*
     * Tell the other side to hang up
     */
    void hangup(int sock);

    /*
     * Listen once
     */
    int listenOnce(Message& msg);

    /*
     * wait until the other side sends confirmation
     */
    void waitConfirm(int sock);
};
#endif   /* ----- #ifndef _BASIC_MESSENGER_H_  ----- */
