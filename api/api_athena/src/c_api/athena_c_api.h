#pragma once

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef int athenaStatus;
typedef double real64;
typedef int    int32;

athenaStatus athena_minbar_label(real64* open, real64* high, real64* low, real64* close, int32 num_bars, // input: all min bars
                          int32* used_time_id, int32 num_ids, // input: index to label
                          real64 ret_thd, // input: return threshold
                          int32 max_stride, // input: max bars to check
                          int32* labels,
                          int32* durations); // output
#ifdef __cplusplus
}
#endif // __cplusplus


