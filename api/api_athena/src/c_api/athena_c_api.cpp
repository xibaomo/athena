#include "athena_c_api.h"
#include <iostream>
using namespace std;

athenaStatus athena_minbar_label(real64* open, real64* high, real64* low, real64* close, int32 num_bars, // input: all min bars
                                 int32* used_time_id, int32 num_ids, // input: index to label
                                 real64 ret_thd, // input: return threshold
                                 int32 max_stride, // input: max bars to check
                                 int32* labels) { // output, length is num_ids

    for (int32 idx = 0; idx < num_ids; idx++) {
        int32 tid = used_time_id[idx];
        real64 p0 = open[tid];
        int32 label = 0;
        for(int j=tid; j < num_bars; j++) {
            if(j - tid > max_stride) break;
            real64 ret_high = high[j]/p0 - 1;
            real64 ret_low  = low[j]/p0 - 1;
            if (ret_high >= ret_thd && ret_low <= -ret_thd) {
//                std::cout << "bar is too long. p0 = " << p0
//                          << ", high = " << high[j] << ", low = " << low[j] << endl;
                break;
            }

            else if (ret_high >= ret_thd) {
                label = 1;
                break;
            } else if (ret_low <= -ret_thd) {
                label = -1;
                break;
            }
        }
        labels[idx] = label;

    }
    return 0;
}
