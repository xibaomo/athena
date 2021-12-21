#include "athena_c_api.h"
#include <iostream>
using namespace std;

athenaStatus athena_minbar_label(real64* open, real64* high, real64* low, real64* close, int32 num_bars, // input: all min bars
                                 int32* used_time_id, int32 num_ids, // input: index to label
                                 real64 ret_thd, // input: return threshold
                                 //real64 ret_thd, // input: deviation of actual return. The true return is (1-dev)*return

                                 int32 max_stride, // input: max bars to check
                                 int32* labels,
                                 int32* durations) { // output, length is num_ids

    real64 dev = 0.06;
    for (int32 idx = 0; idx < num_ids; idx++) {
        int32 tid = used_time_id[idx];
        real64 p0 = open[tid];
        int32 label = 0;
        int32 dur = -1;
        for(int j=tid; j < num_bars; j++) {
            if(j - tid > max_stride) break;
            dur = j - tid;
            real64 ret_high = high[j]/p0 - 1;
            real64 ret_low  = low[j]/p0 - 1;
            // first check if we can take profit in this bar
            // actual return if profit
            real64 rhp = ret_high*(1-dev);
            real64 rlp = ret_low*(1-dev);
            if (rhp >= ret_thd && rlp <= -ret_thd) {
                label = 2;
                break;
            }
            else if(rhp >= ret_thd) {
                label = 1;
                break;
            }
            else if (rlp <= -ret_thd) {
                label = -1;
                break;
            }

            // Next check if we stop loss in this bar
            // actual return if lose
            real64 rhl = ret_high*(1+dev);
            real64 rll = ret_low*(1+dev);
            if (rhl >= ret_thd && rll <= -ret_thd) {
//                std::cout << "bar is too long. p0 = " << p0
//                          << ", high = " << high[j] << ", low = " << low[j] << endl;
                label = 2; // hard to tell
                break;
            }

            else if (rhl >= ret_thd) {
                label = 1;
                break;
            } else if (rll <= -ret_thd) {
                label = -1;
                break;
            }
        }
        if (label==0) {
            std::cerr << "Label should not be 0!!!" << std::endl;
        }
        labels[idx] = label;
        durations[idx] = dur;

    }
    return 0;
}
