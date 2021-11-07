/*
 * =====================================================================================
 *
 *       Filename:  create_mbp.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/20/2019 06:44:06 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "create_mbp.h"
#include "mbtconf.h"
#include "minbar_predictor/ma_pred/createMAPredictor.h"
using namespace std;

MinBarBasePredictor*
createMBPredictor(MBP_Type mt, const String& cfg)
{
    MinBarBasePredictor* p (nullptr);
    switch(mt) {
        case MBP_Type::MA_PRED:
            p = (MinBarBasePredictor*)createMAPredictor(cfg);
            break;
        default:
            Log(LOG_FATAL) << "Unrecognized predictor type: " + to_string((int)mt);
            break;
    }

    return p;
}
