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
#include <unordered_map>
#include <typeinfo>
#include "types.h"
namespace logging = boost::log;
enum class Log_Level : int {
    LOG_FATAL = 0,
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
    LOG_VERBOSE,
    LOG_DEBUG
};
extern Log_Level LOG_FATAL;
extern Log_Level LOG_WARNING;
extern Log_Level LOG_ERROR;
extern Log_Level LOG_INFO;
extern Log_Level LOG_VERBOSE;
extern Log_Level LOG_DEBUG;

class Logger
{
private:
    Log_Level m_curLvl;
    std::unordered_map<int, decltype(logging::trivial::info)> m_lvlDict;

public:
    Logger()
    {
        m_lvlDict[(int)Log_Level::LOG_FATAL] = logging::trivial::fatal;
        m_lvlDict[(int)Log_Level::LOG_ERROR] = logging::trivial::error;
        m_lvlDict[(int)Log_Level::LOG_WARNING] = logging::trivial::warning;
        m_lvlDict[(int)Log_Level::LOG_INFO] = logging::trivial::info;
        m_lvlDict[(int)Log_Level::LOG_VERBOSE] = logging::trivial::debug;
        m_lvlDict[(int)Log_Level::LOG_DEBUG] = logging::trivial::trace;
        setLogLevel(LOG_INFO);
    }

    static Logger& getInstance()
    {
        static Logger _inst;
        return _inst;
    }

    void setLogLevel(Log_Level lvl)
    {
        auto loglvl = m_lvlDict[(int)lvl];

        logging::core::get()->set_filter(logging::trivial::severity >= loglvl);
    }

    Logger& operator ()(Log_Level level)
    {
        m_curLvl = level;
        return *this;
    }

    template <typename T>
    Logger& operator << (const T& msg) {
        auto loglvl = m_lvlDict[(int)m_curLvl];
        BOOST_LOG_STREAM_WITH_PARAMS(::boost::log::trivial::logger::get(), \
        (::boost::log::keywords::severity = loglvl)) << msg;

        return *this;
    }
};

extern Logger Log;
#endif   /* ----- #ifndef _BASICS_LOG_H_  ----- */
