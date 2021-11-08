#pragma once
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include <python3.8/Python.h>

class MinbarPyPredictor : public MinBarBasePredictor {
protected:
    PyObject* m_mod;
public:
    MinbarPyPredictor();
    virtual ~MinbarPyPredictor() {
        if(m_mod) {
            Py_DECREF(m_mod);
        }
    }

    void setPredictorFile(const String& pf) override;

    void prepare();
    FXAct predict(real64 new_open);

    /////////////////// internal function /////////////////////
    void loadMinbarsToPredictor();
    void addMinbarToPredictor(const MinBar& mb);

};


