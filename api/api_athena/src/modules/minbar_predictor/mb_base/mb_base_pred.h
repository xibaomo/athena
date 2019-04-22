/*
 * =====================================================================================
 *
 *       Filename:  mb_base_pred.h
 *
 *    Description:  This file defines the base class of various min bar predictors
 *
 *        Version:  1.0
 *        Created:  04/20/2019 06:20:29 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MINBAR_BASE_PREDICTOR_H_
#define  _MINBAR_BASE_PREDICTOR_H_

#include <vector>
#include "fx_action/fx_action.h"
struct MinBar {
    String time;
    real32 open;
    real32 high;
    real32  low;
    real32 close;
    int32  tickvol;

    MinBar(String p_time,real32 p_open,real32 p_high,real32 p_low,real32 p_close,int32 p_tickvol):
        time(std::move(p_time)),open(p_open),high(p_high),low(p_low),close(p_close),tickvol(p_tickvol)
        {
            ;
        }
};
class MinBarBasePredictor {
protected:
    std::vector<MinBar>* m_allMinBars;
    MinBarBasePredictor() {;}
public:
    virtual ~MinBarBasePredictor() {;}

    virtual void loadAllMinBars(std::vector<MinBar>* bars) {
        m_allMinBars = bars;
    }

    virtual void prepare() = 0;

    virtual FXAction predict() = 0;
};
#endif   /* ----- #ifndef _MINBAR_BASE_PREDICTOR_H_  ----- */
