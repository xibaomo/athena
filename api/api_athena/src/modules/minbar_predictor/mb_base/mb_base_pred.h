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
struct MinBar;
class MinBarBasePredictor {
protected:
    std::vector<MinBar>* m_allMinBars;
    MinBarBasePredictor() {;}
public:
    virtual ~MinBarBasePredictor() {;}

    void loadAllMinBars(std::vector<MinBar>* bars) {
        m_allMinBars = bars;
    }
};
#endif   /* ----- #ifndef _MINBAR_BASE_PREDICTOR_H_  ----- */
