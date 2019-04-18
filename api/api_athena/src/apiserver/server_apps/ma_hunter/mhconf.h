/*
 * =====================================================================================
 *
 *       Filename:  mhconf.h
 *
 *    Description:  This file defines MA hunter config
 *
 *        Version:  1.0
 *        Created:  04/18/2019 02:35:52 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MA_HUNTER_CONFIG_H_
#define  _MA_HUNTER_CONFIG_H_

#include "basics/baseconf.h"
class MahuntConfig : public BaseConfig {
protected:
    MahuntConfig() {;}
public:
    virtual ~MahuntConfig() {;}
    static MahuntConfig& getInstance() {
        static MahuntConfig _ins;
        return _ins;
    }
};
#endif   /* ----- #ifndef _MA_HUNTER_CONFIG_H_  ----- */
