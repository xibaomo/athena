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
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include "mhconf.h"
#include "ma_cal.h"
class  MAHunter : public MinBarBasePredictor {
protected:

    std::vector<real32> m_median;
    std::vector<real32> m_ma;

    MACalculator* m_maCal;
    MahuntConfig* m_config;
    MAHunter(const String& cfg);
public:
    virtual ~MAHunter() {;}
    static MAHunter& getInstance(const String& cfg) {
        static MAHunter _ins(cfg);
        return _ins;
    }

    void prepare();

    FXAction predict();

    void compLWMA();
    real32 compALWMA(int lookback, size_t idx);
};
#endif   /* ----- #ifndef _SERVER_MA_HUNTER_H_  ----- */
