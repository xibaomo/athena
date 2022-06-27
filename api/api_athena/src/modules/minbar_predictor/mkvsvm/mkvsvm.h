#pragma once
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include "minbar_predictor/py_pred/py_pred.h"

class MkvSvmPredictor : public MinBarBasePredictor {
private:
    PyObject* m_mod;

    String m_predConfigFile;

    MinbarPyPredictor m_pyPredictor;
public:
    MkvSvmPredictor();
    ~MkvSvmPredictor() {
        if ( m_mod ) {
            Py_DECREF(m_mod);
        }
    }

    void loadConfig();

    void setPredictorFile(const String& path, const String& pf) override {
        m_predConfigFile = pf;
    }
    void prepare() override {
        loadConfig();
        m_pyPredictor.loadAllMinBars(m_allMinBars);
        m_pyPredictor.prepare();
    }
    void appendMinbar(const MinBar& mb) override {
        m_allMinBars->push_back(mb);

        m_pyPredictor.appendMinbar(mb);
    }

    int predict(const String& ts, real64 new_open) override {
        return m_pyPredictor.predict(ts, new_open);
    }

    real64 getReturn() override;
};
