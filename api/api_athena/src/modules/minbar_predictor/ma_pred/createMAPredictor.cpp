/*
 * =====================================================================================
 *
 *       Filename:  createMAPredictor.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/26/2019 02:14:41 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "createMAPredictor.h"
#include "mpconf.h"
#include "minbar_predictor/ma_pred/ma_pred_base/ma_pred_base.h"
#include "minbar_predictor/ma_pred/grad_hunter/ma_grad_hunter.h"
#include "minbar_predictor/ma_pred/crosser/ma_crosser.h"

MABasePredictor*
createMAPredictor(const String& cf)
{
    MABasePredictor* mp(nullptr);
    MPConfig* cfg = &MPConfig::getInstance();
    cfg->loadConfig(cf);

    MA_TYPE mt = cfg->getMAType();
    MACalculator*  macal = createMACalculator(mt);

    switch(cfg->getMAPredictorType()) {
        case MP_Type::MA_GRAD_HUNT:
            mp = &MAHunter::getInstance(cf,macal);
            break;
        case MP_Type::LS_MA_CROSS:
            mp = &MACrosser::getInstance(cf,macal);
            break;
        default:
            Log(LOG_FATAL) << "MA predictor type not supported";
            break;
    }

    return mp;
}
