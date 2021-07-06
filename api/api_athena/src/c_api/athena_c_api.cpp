#include "athena_c_api.h"
#include "basics/log.h"
#include "client_apps/client_predictor/client_predictor.h"
#include "pyhelper.hpp"
#include "basics/utils.h"
#include <iostream>
using namespace std;
using namespace athena;

String _getModelFile(String qmdlFile)
{
    String athena_install = String(getenv("ATHENA_INSTALL"));
    String modulePath = athena_install + "/api/release/scripts/";

    PyEnviron::getInstance().appendSysPath(modulePath);

    CPyObject qmdlParserMod = PyImport_ImportModule("qmdl_parser");
    CPyObject pDict = PyModule_GetDict(qmdlParserMod);
    CPyObject pFunc = PyDict_GetItemString(pDict, "getValueByKey");

    CPyObject val = PyObject_CallFunction(pFunc, "ss", "EXTERNAL_MODEL/MODEL_PATH",qmdlFile.c_str());

    auto v = String(PyBytes_AsString(val));
    return v;
}

String getModelFile(String qmdlFile)
{
    String athena_install = String(getenv("ATHENA_INSTALL"));
    String parser = athena_install + "/api/release/scripts/qmdl_parser.py";
    String cmd = parser + " " + qmdlFile;

    String modelfile = execSysCall_block(cmd);
    return modelfile;
}

Status athena_load(void** pc, char* qmdl_config_file)
{
    *pc = new ClientPredictor();
    ClientPredictor* app = (ClientPredictor*)*pc;

    String model_file = getModelFile(String(qmdl_config_file));

    app->prepare(model_file);

    return 0;
}

Status athena_unload(void** pc)
{
    delete (ClientPredictor*)*pc;
    *pc = nullptr;

    return 0;
}

Status comp_prediction(void* pc,
        Real* features,
        const Uint n_samples,
        const Uint n_features,
        Real* bias)
{
    ClientPredictor* app = (ClientPredictor*)pc;
    app->sendFeatures(features, n_samples, n_features);
    app->getResult(bias, n_samples);

    return 0;
}
