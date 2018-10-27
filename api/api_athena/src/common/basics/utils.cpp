/*
 * =====================================================================================
 *
 *       Filename:  utils.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 02:33:24 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "utils.h"
#include <memory>
#include <array>
#include <chrono>
#include <thread>
#include <unistd.h>
using namespace std;

String
execSysCall(const String& cmd)
{
    array<char, 128> buffer;
    String result;
    shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if ( !pipe )
        throw runtime_error("popen() failed");

    while ( !feof(pipe.get()) ) {
        if ( fgets(buffer.data(), 128, pipe.get()) != nullptr )
            result += buffer.data();
    }

    result.erase(result.find('\n'));

    return result;
}

void sleepMilliSec(int num_ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(num_ms));
}

String
getHostName()
{
    char hn[128];
    int err = gethostname(hn,128);

    if (err<0)
        throw runtime_error("cannot get host name");

    return String(hn);
}