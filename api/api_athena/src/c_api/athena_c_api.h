#ifndef _CAPI_ATHENA_H_
#define _CAPI_ATHENA_H_

typedef int Status; /* 0 - success. Non-zero: failure */
typedef float Real;
typedef unsigned int Uint;

Status athena_load(void** athena_engine, char* model_file);

Status comp_prediction(void* athena_engine,
                       Real* features,
                       const Uint n_samples,
                       const Uint n_features,
                       Real* bias);

Status athena_unload(void** metis_engine);

#endif
