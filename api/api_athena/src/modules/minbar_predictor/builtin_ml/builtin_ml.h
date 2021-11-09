#pragma once
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include "minbar_predictor/py_pred/py_pred.h"
#include "minbar_tracker/mbtconf.h"

class BMLConfig {
protected:
    const String BMLROOT = "MINBAR_TRACKER/BUILTIN_ML/";
    MbtConfig* m_cfg;
public:
    BMLConfig(MbtConfig* cfg) : m_cfg(cfg) {;}

    String getModelFile() {
        return m_cfg->getKeyValue<String>(BMLROOT + "MODEL_FILE");
    }
};

class BuiltinMLPredictor : public MinBarBasePredictor {
private:
    BMLConfig* m_cfg;
    MinbarPyPredictor m_pyPredictor;
public:
    BuiltinMLPredictor(MbtConfig* cfg) : m_cfg(cfg) {;}
    ~BuiltinMLPredictor()  = default;

    void prepare() override {
        m_pyPredictor.loadAllMinBars(m_allMinBars);
        m_pyPredictor.prepare();
    }
    void appendMinbar(const MinBar& mb) override {
        m_pyPredictor.appendMinbar(mb);
    }

    FXAct predict(real64 new_open) override {
        return m_pyPredictor.predict(new_open);
    }
};
