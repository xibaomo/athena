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
#include <locale>
#include <string>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace std;
namespace athena{
String
execSysCall_block(const String& cmd)
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
    int err = gethostname(hn, 128);

    if ( err<0 )
        throw runtime_error("cannot get host name");

    return String(hn);
}

vector<String>
splitString(const String& str, const String delimiters)
{
    vector<String> res;
    boost::split(res, str, [delimiters](char c) {
                 for (auto a : delimiters) {
                    if (c == a) return true;
                 }
                 return false;});
    return res;
}

bool
compareStringNoCase(const String& str1, const String& str2)
{
    return boost::iequals(str1,str2);
}

String
getStringFromPyobject(CPyObject& pyobj)
{
    PyObject* objrepr = PyObject_Repr(pyobj.getObject());
    if (!objrepr) {
        throw runtime_error("Failed to get string from pyobject");
    }

    const char* cp = PyString_AsString(objrepr);
    Py_XDECREF(objrepr);

    String s = String(cp);
    return s.substr(1,s.size()-2);
}

int
getIntFromPyobject(CPyObject& pyobj)
{
    PyObject* objrepr = PyObject_Repr(pyobj.getObject());
    if (!objrepr) {
        throw runtime_error("Failed to get string from pyobject");
    }

    const char* cp = PyString_AsString(objrepr);

    if (!cp) throw runtime_error("Get Null from PyString_AsString");
    int p = stoi(String(cp));
    Py_XDECREF(objrepr);
    return p;
}
NonBlockSysCall* gNBSysCall = &NonBlockSysCall::getInstance();

long
getTimeDiffInMin(const String& st1, const String& st2)
{
    boost::posix_time::ptime t1(boost::posix_time::time_from_string(st1));
    boost::posix_time::ptime t2(boost::posix_time::time_from_string(st2));

    boost::posix_time::time_duration td = t1 - t2;
    long diffmin = td.total_seconds()/60;

    return diffmin;
}

String
convertTimeString(const String& timeStr, const String& fmt)
{
    boost::posix_time::ptime t(boost::posix_time::time_from_string(timeStr));
    std::stringstream stream;
    boost::posix_time::time_facet* facet = new boost::posix_time::time_facet();
    facet->format(fmt.c_str());
    stream.imbue(std::locale(std::locale::classic(),facet));
    stream << t;

    String resStr = stream.str();
//    delete facet;
    return resStr;
}
}
