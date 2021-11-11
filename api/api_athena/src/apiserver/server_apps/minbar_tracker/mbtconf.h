/*
 * =====================================================================================
 *
 *       Filename:  mbtconf.h
 *
 *    Description:  Config class for min bar tracker
 *
 *        Version:  1.0
 *        Created:  04/20/2019 06:34:04 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#pragma once

#include "basics/baseconf.h"

const String MBT_ROOT = "MINBAR_TRACKER/";
class MbtConfig : public BaseConfig {
protected:

public:
    MbtConfig() {;}
    virtual ~MbtConfig() {;}

    int getPredictorType() {
        return getKeyValue<int>(MBT_ROOT + "PREDICTOR_TYPE");
    }

    String getBMLConfigFile() {
        return getKeyValue<String>(MBT_ROOT + "BUILTIN_ML/CONFIG_FILE");
    }

    String getCustomPyFile() {
        return getKeyValue<String>(MBT_ROOT + "CUSTOM_PY/PY_FILE");
    }

};
