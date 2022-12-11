#include "athn.h"
#include <omp.h>
using namespace std;
athnStatus athn_label_minbars(real64* open, real64* high, real64* low, uint64* secs, int32 num_bars, // input: all min bars
                              uint64* minbar_ids, int32 num_minbar_ids, // input: index to label
                              real64 ret_thd, // input: return threshold
                              int32 ndays, // input: position lifetime
                              int32* labels) {
// secs is the elapsed time in second relative to the first min bar

    uint64 pos_lifetime = ndays*3600*24;
//    omp_set_num_threads(32);
//#pragma omp parallel for
    for(int32 idx = 0; idx < num_minbar_ids; idx++) {
        int32 tid = minbar_ids[idx];
        real64 p0 = open[tid];
        uint64 t0 = secs[tid];
        int32 label = 0;
        for(int32 j=tid; j < num_bars; j++) {
            real64 rh = high[j]/p0-1.f;
            real64 rl = low[j]/p0-1.f;

            if (rh > ret_thd && rl < -ret_thd) {
                label = 3;
                break;
            } else if(rh >= ret_thd) {
                label = 1;
                break;
            } else if(rl <= -ret_thd) {
                label = 2;
                break;
            } else {
                uint64 dt = secs[j] - t0;
                if (dt >= pos_lifetime) {// position's time is up.
                    if (rl > 0) label = 1;
                    if (rh < 0) label = 2;
                    break;
                }
            }
        }
        labels[idx] = label;
    }
    return 0;
}
