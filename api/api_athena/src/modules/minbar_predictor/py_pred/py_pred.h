#pragma once
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include <python3.8/Python.h>

class MinbarPyPredictor : public MinBarBasePredictor {
protected:
    PyObject* m_mod;
public:
    MinbarPyPredictor();
    virtual ~MinbarPyPredictor() {
        finish();
        if(m_mod) {
            Py_DECREF(m_mod);
        }
    }

    void setPredictorFile(const String& path, const String& pf) override;

    void prepare();
    void appendMinbar(const MinBar& mb) override;
    int predict(const String& time_str, real64 new_open);
    void finish();

    /////////////////// internal function /////////////////////
    void loadMinbarsToPredictor();


};


