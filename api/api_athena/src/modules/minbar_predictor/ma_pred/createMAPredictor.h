/*
 * =====================================================================================
 *
 *       Filename:  createMAPredictor.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/26/2019 02:12:57 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _CREATE_MA_PREDICTOR_H_
#define  _CREATE_MA_PREDICTOR_H_

#include "minbar_predictor/ma_pred/ma_pred_base/ma_pred_base.h"

MABasePredictor* createMAPredictor(const String& cf);

#endif   /* ----- #ifndef _CREATE_MA_PREDICTOR_H_  ----- */
