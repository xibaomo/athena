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
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include "basics/log.h"
#include "types.h"

/*-----------------------------------------------------------------------------
 *  Execute system call by popen and return result as a string
 *-----------------------------------------------------------------------------*/
String execSysCall_block(const String& cmd);

class NonBlockSysCall {
private:
    std::vector<FILE*> m_fhs;
    char m_buffer[1024];
    NonBlockSysCall(){;}
public:

    static NonBlockSysCall& getInstance() {
        static NonBlockSysCall _inst;
        return _inst;
    }
    void exec(const String& cmd) {
        FILE* fh = popen(cmd.c_str(), "r");
        int d = fileno(fh);
        fcntl(d, F_SETFL, O_NONBLOCK);
        m_fhs.push_back(fh);
    }
   virtual ~NonBlockSysCall() {
       for ( auto fh: m_fhs) pclose(fh);
   }
//    bool checkFinished() {
//        int d = fileno(m_fh);
//        ssize_t r = read(d, m_buffer, 1024);
//        if ( r == -1 && errno == EAGAIN ) {
//            //Log(LOG_VERBOSE) << m_cmd + " not finished";
//            return false;
//        } else if (r > 0) {
//            return true;
//        } else
//            Log(LOG_ERROR) << "Pipe closed";
//        return false;
//    }

    String getResult() {
        String res(m_buffer);
        return res;
    }
};

extern NonBlockSysCall* gNBSysCall;
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
splitString(const String& str, const String delimiters=":");

/**
 * Case-insensitive string comparison
 */

bool
compareStringNoCase(const String& str1, const String& str2);

class Timer {
protected:
    std::chrono::time_point<std::chrono::system_clock> m_start;
public:
    Timer() { m_start = std::chrono::system_clock::now();}
    double getElapsedTime() {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = now - m_start;
        return elapsed.count();
    }

};
#endif   /* ----- #ifndef _BASIC_UTILS_H_  ----- */
