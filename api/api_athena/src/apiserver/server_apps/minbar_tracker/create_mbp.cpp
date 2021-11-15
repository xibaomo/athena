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
#include "basics/utils.h"
#include "minbar_predictor/builtin_ml/builtin_ml.h"
#include "minbar_predictor/py_pred/py_pred.h"
using namespace std;
using namespace athena;

MinBarBasePredictor*
createMBPredictor(int type,MbtConfig* cfg)
{
    MinBarBasePredictor* p (nullptr);
    switch(type) {
    case 0: {
        p = new BuiltinMLPredictor();
        String cf = cfg->getKeyValue<String>("MINBAR_TRACKER/BUILTIN_ML/CONFIG_FILE");
        p->setPredictorFile("",cf);
        break;
    }
    case 1: {
        p = new MinbarPyPredictor();
        Log(LOG_FATAL) << "Customized python predictor not supported";
        break;
    }
    default:
        Log(LOG_FATAL) << "Predictor type not supported: " + to_string(type);
        break;
    }

    return p;
}
