#pragma once

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef int athnStatus;
typedef double real64;
typedef int    int32;
typedef unsigned long   uint64;

athnStatus athn_label_minbars(real64* open, real64* high, real64* low,
                              uint64* secs, //
                              int32 num_bars, // input: number of min bars
                          int32* minbar_ids, int32 num_minbar_ids, // input: index to label
                          real64 ret_thd, // input: return threshold
                          int32 ndays, // input: position lifetime
                          int32* labels); // output
#ifdef __cplusplus
}
#endif // __cplusplus


