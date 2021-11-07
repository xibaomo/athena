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
    MbtConfig() {;}
public:
    virtual ~MbtConfig() {;}

    static MbtConfig& getInstance() {
        static MbtConfig _ins;
        return _ins;
    }

    MBP_Type getMinBarPredictorType() {
        return (MBP_Type)getKeyValue<int>(MBT_ROOT + "PREDICTOR_TYPE");
    }

    String getHistoryMinBarFile() {
        return getKeyValue<String>(MBT_ROOT + "HISTORY_BAR_FILE");
    }
};
#endif   /* ----- #fndef _MIN_BAR_CONFIG_H_  ----- */
