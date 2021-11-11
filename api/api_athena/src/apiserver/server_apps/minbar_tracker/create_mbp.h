/*
 * =====================================================================================
 *
 *       Filename:  create_mbp.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/20/2019 06:38:10 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MIN_BAR_CREATE_H_
#define  _MIN_BAR_CREATE_H_
#include "basics/types.h"
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include "minbar_tracker/mbtconf.h"

MinBarBasePredictor*
createMBPredictor(int pred_type,MbtConfig* cfg);

#endif   /* ----- #ifndef _MIN_BAR_CREATE_H_  ----- */
