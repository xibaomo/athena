//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
#ifndef _PREDICTOR_PREDICTOR_H_
#define _PREDICTOR_PREDICTOR_H_

enum class PrdAction {
    SETUP = 10,
    PREDICT = 11,
    RESULT
};

enum class EngineType : int {
    REGRESSOR = 0,
    CLASSIFIER
};

enum class EngineCoreType: int {
    SVM = 0,
    RANDOMFOREST,
    DNN
};
#endif
