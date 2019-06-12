/*
 * =====================================================================================
 *
 *       Filename:  generalconf.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/19/2019 02:23:41 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _GENERAL_CONFIG_H_
#define  _GENERAL_CONFIG_H_

#include "basics/baseconf.h"
#include "basics/types.h"
#include "basics/log.h"
const String GENERAL_ROOT = "GENERAL/";

class GeneralConfig : public BaseConfig {
protected:
    GeneralConfig() {;}
public:
    virtual ~GeneralConfig() {;}

    static GeneralConfig& getInstance() {
        static GeneralConfig _ins;
        return _ins;
    }

    AppType getAppType() {
        return (AppType)getKeyValue<int>(GENERAL_ROOT+ "APPLICATION");
    }

    LogLevel getLogLevel() {
        return (LogLevel)getKeyValue<int>(GENERAL_ROOT + "LOG_LEVEL");
    }

    int getPort() {
        return getKeyValue<int>(GENERAL_ROOT + "PORT");
    }
};
#endif   /* ----- #ifndef _GENERAL_CONFIG_H_  ----- */
