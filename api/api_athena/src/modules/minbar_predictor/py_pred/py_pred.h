#pragma once
#include "minbar_predictor/mb_base/mb_base_pred.h"

class MinbarPyPredictor : public MinBarBasePredictor {
public:
    MinbarPyPredictor() = default;
    virtual ~MinbarPyPredictor() = default;

    void prepare() {;}
    FXAct predict() { return FXAct::NOACTION; }
};


