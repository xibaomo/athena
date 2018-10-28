/*
 * =====================================================================================
 *
 *       Filename:  log.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 12:26:31 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "log.h"
#define BOOST_LOG_DYN_LINK 1
Logger Log = Logger::getInstance();

Log_Level LOG_FATAL = Log_Level::LOG_INFO;
Log_Level LOG_ERROR = Log_Level::LOG_ERROR;
Log_Level LOG_WARNING = Log_Level::LOG_WARNING;
Log_Level LOG_INFO = Log_Level::LOG_INFO;
Log_Level LOG_DEBUG = Log_Level::LOG_DEBUG;
Log_Level LOG_VERBOSE = Log_Level::LOG_VERBOSE;
