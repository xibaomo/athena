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

#ifndef  _MIN_BAR_CONFIG_H_
#define  _MIN_BAR_CONFIG_H_

#include "basics/baseconf.h"
#include "create_mbp.h"

const String MBT_ROOT = "MINBAR_TRACKER/";
class MbtConfig : public BaseConfig {
protected:

public:
    MbtConfig() {;}
    virtual ~MbtConfig() {;}

    String getPredictorFile() {
        return getKeyValue<String>(MBT_ROOT + "PREDICTOR_FILE");
    }

    String getFeatureConfigFile() {
        return getKeyValue<String>(MBT_ROOT + "FEATURE_CONFIG_FILE");
    }
};
#endif   /* ----- #fndef _MIN_BAR_CONFIG_H_  ----- */
