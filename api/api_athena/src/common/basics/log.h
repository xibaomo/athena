/*
 * =====================================================================================
 *
 *       Filename:  log.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 12:26:22 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BASICS_LOG_H_
#define  _BASICS_LOG_H_

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <unordered_map>
#include <typeinfo>
#include "types.h"

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;

enum LogLevel {
    LOG_FATAL = 0,
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
    LOG_VERBOSE,
    LOG_DEBUG
};

class LogStream
{
private:
    LogLevel m_curLevel;
    std::unordered_map<int, decltype(logging::trivial::info)> m_lvlDict;
    LogStream()
    {
        m_lvlDict[(int)LogLevel::LOG_FATAL] = logging::trivial::fatal;
        m_lvlDict[(int)LogLevel::LOG_ERROR] = logging::trivial::error;
        m_lvlDict[(int)LogLevel::LOG_WARNING] = logging::trivial::warning;
        m_lvlDict[(int)LogLevel::LOG_INFO] = logging::trivial::info;
        m_lvlDict[(int)LogLevel::LOG_VERBOSE] = logging::trivial::debug;
        m_lvlDict[(int)LogLevel::LOG_DEBUG] = logging::trivial::trace;

//        logging::add_file_log(keywords::file_name = "athena.log",
//                              keywords::format = "[%TimeStamp%]: %Message%");
    }
public:
    static LogStream& getInstance()
    {
        static LogStream _instance;
        return _instance;
    }

    void setLogLevel(LogLevel lvl)
    {
        auto loglvl = m_lvlDict[(int)lvl];

        logging::core::get()->set_filter(logging::trivial::severity >= loglvl);
    }

    void setCurLogLevel(LogLevel lvl)
    {
        m_curLevel = lvl;
    }

    template <typename T>
    LogStream& operator << (const T& msg)
    {
        auto loglvl = m_lvlDict[(int)m_curLevel];
        BOOST_LOG_STREAM_WITH_PARAMS(::boost::log::trivial::logger::get(), \
                                     (::boost::log::keywords::severity = loglvl)) << msg;

        return *this;
    }
};

class Logger
{
public:

    Logger(LogLevel lvl)
    {
        auto& logger = LogStream::getInstance();
        logger.setCurLogLevel(lvl);
    }

    static void setLogLevel(LogLevel lvl)
    {
        auto& logger = LogStream::getInstance();
        logger.setLogLevel(lvl);
    }

    template <typename T>
    Logger& operator << (const T& msg)
    {
        auto& logger = LogStream::getInstance();
        logger << msg;
        return *this;
    }
};

typedef Logger Log;
//extern Logger Log;
#endif   /* ----- #ifndef _BASICS_LOG_H_  ----- */
