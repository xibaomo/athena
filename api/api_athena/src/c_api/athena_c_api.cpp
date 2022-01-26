#include "athena_c_api.h"
#include "basics/utils.h"
#include <iostream>
#include <math.h>
using namespace std;
using namespace athena;

// get point unit of the symbol
static real64 getPoint3or5(real64 v) {
    // first check if 1e-3
    real64 vv = v*1e3;
    int rv = std::round(vv);
    if (std::abs(vv - rv) < 1e-5)
        return 1e-3;

    // second if 1e-5
    vv = v*1e5;
    rv = std::round(vv);
    if(std::abs(vv  - rv) < 1e-5) {
        return 1e-5;
    }

    return -1;
}

athenaStatus athena_minbar_label(real64* open, real64* high, real64* low, real64* close, real64* spread, int32 num_bars, // input: all min bars
                                 int32* used_time_id, int32 num_ids, // input: index to label
                                 real64 ret_thd, // input: return threshold
                                 real64 profit_return_ratio, // the true return of a profit return is ret*thd*profit_return_ratio

                                 int32 max_stride, // input: max bars to check
                                 int32* labels,
                                 int32* durations) {
    const real64 ONE = 1.f;
    real64 pv = getPoint3or5(open[0]);
    std::cerr<<"Point unit of symbol: " << pv << endl;

    size_t bnb = 0; // break neither of bounds
    size_t bbb = 0; // break both bounds
//    std::cout<<"Max stride: " << max_stride << endl;
//
//    std::cout<<"open hash: " << hasharray(open,num_bars) << endl;
//    std::cout<<"high hash: " << hasharray(high,num_bars) << endl;
//    std::cout<<"low hash: " << hasharray(low,num_bars) <<endl;
//    std::cout<<"close hash: " << hasharray(close,num_bars) << endl;
//    std::cout<<"spread hash: " << hasharray(spread,num_bars) << endl;

    real64 true_ratio = profit_return_ratio;
    for(int32 idx = 0; idx < num_ids; idx++) {
        int32 tid = used_time_id[idx];
        real64 p0_buy = open[tid] + pv*spread[tid];
        real64 p0_sell = open[tid];
        int32 label = 0;
        int32 dur = -1;
        for(int j=tid; j < num_bars; j++) {
            if (j-tid > max_stride) break;
            dur = j - tid;
            real64 ret_buy = high[j]/p0_buy - ONE;
            real64 ret_sell = (low[j]+pv*spread[j])/p0_sell - ONE;

            real64 true_ret_buy = ret_buy * true_ratio;
            real64 true_ret_sell = ret_sell * true_ratio;
            // first check if we can take profit
            if (true_ret_buy >= ret_thd && true_ret_sell <= -ret_thd) {
                label = 2;
                bbb++;
                break;
            } else if (true_ret_buy >= ret_thd) {
                label = 1;
                break;
            } else if (true_ret_sell <= -ret_thd) {
                label = -1;
                break;
            }

            // next check if stop loss

        }
        if (label==0) {
//            std::cerr << "Label should not be 0!!!" << std::endl;
            bnb++;
        }
        labels[idx] = label;
        durations[idx] = dur;
    }

//    cout<<"hash labels: " << hasharray(labels,num_ids) << endl;
    std::cerr << "Break both bounds (threshold is too small): " << bbb << endl;
    std::cerr << "Break neither of bounds (threshold is too large): " << bnb << endl;

    return 0;
}

athenaStatus __athena_minbar_label(real64* open, real64* high, real64* low, real64* close, int32 num_bars, // input: all min bars
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
            } else if(rhp >= ret_thd) {
                label = 1;
                break;
            } else if (rlp <= -ret_thd) {
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
