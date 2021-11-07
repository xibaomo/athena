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
#include "basics/utils.h"
using namespace std;
using namespace athena;

MinBarBasePredictor*
createMBPredictor( const String& pf)
{
    MinBarBasePredictor* p (nullptr);

    String ext = getFileExt(pf);
    if (ext == ".py"){
        p = nullptr;
    } else {
        Log(LOG_FATAL) << "Predictor file not supported: " + pf;
    }

    return p;
}
