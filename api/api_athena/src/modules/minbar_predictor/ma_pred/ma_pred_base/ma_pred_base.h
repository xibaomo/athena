/*
 * =====================================================================================
 *
 *       Filename:  ma_base_pred.h
 *
 *    Description:  Base class of MA predictor
 *
 *        Version:  1.0
 *        Created:  04/28/2019 03:17:46 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MA_PRED_BASE_H_
#define  _MA_PRED_BASE_H_
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include "minbar_predictor/ma_pred/ma_cal/ma_cal.h"

class MABasePredictor : public MinBarBasePredictor{
protected:
    MACalculator* m_maCal;
    MABasePredictor(MACalculator* cal) : m_maCal(cal) {;}
public:
    virtual ~MABasePredictor() {;}
};
#endif   /* ----- #ifndef _MA_PRED_BASE_H_  ----- */

