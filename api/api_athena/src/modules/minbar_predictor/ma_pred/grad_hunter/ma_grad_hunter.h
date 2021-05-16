/*
 * =====================================================================================
 *
 *       Filename:  ma_hunter.h
 *
 *    Description:  This class defines MA hunter
 *
 *        Version:  1.0
 *        Created:  04/18/2019 03:03:35 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _SERVER_MA_HUNTER_H_
#define  _SERVER_MA_HUNTER_H_

#include "messenger/msg.h"
#include "minbar_predictor/ma_pred/ma_pred_base/ma_pred_base.h"
#include "mghconf.h"
#include "minbar_predictor/ma_pred/ma_cal/ma_cal.h"

class  MAHunter : public MABasePredictor {
protected:

    std::vector<real32> m_median;
    std::vector<real32> m_ma;

    std::vector<real32> m_records;

    MahuntConfig* m_config;
    MAHunter(const String& cf, MACalculator* cal);
public:
    virtual ~MAHunter();
    static MAHunter& getInstance(const String& cf,MACalculator* cal) {
        static MAHunter _ins(cf,cal);
        return _ins;
    }

    void dumpRecords();
    void prepare();

    FXAct predict();

    void compLWMA();
    real32 compALWMA(int lookback, size_t idx);

    int findNearestTurnPoint(std::vector<real64>& curve);
};
#endif   /* ----- #ifndef _SERVER_MA_HUNTER_H_  ----- */
