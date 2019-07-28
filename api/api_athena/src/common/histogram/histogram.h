/*
 * =====================================================================================
 *
 *       Filename:  histogram.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  07/27/2019 01:08:02 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _COMMON_HISTOGRAM_H_
#define  _COMMON_HISTOGRAM_H_

#include "basics/types.h"
#include "basics/log.h"
#include <vector>
#include <unordered_map>

template <typename T>
int bisec_search_bin(T x, std::vector<T>& bins)
{
    size_t low = 0, high = bins.size()-1;
    if ( x < bins[0] || x > bins.back()) return -1;
    while ( 1 ) {
        if ( high-low == 1) return low;
        size_t idx = (low+high)/2;
        if ( bins[idx] <= x && bins[idx+1] > x )
            return idx;
        if ( x < bins[idx] )
            high = idx;
        if ( x > bins[idx+1] )
            low = idx;
    }

    return -1;
}

template <typename T>
std::vector<size_t> histogram(std::vector<T>& data, std::vector<T>& bins)
{
    //num_bins = bins.size()-1
    std::vector<size_t> contents(bins.size()-1, 0);
    for ( auto& v : data ) {
        int idx = bisec_search_bin(v, bins);
        if ( idx < 0 ) {
            Log(LOG_FATAL) << "Failed to find bin containing " + std::to_string(v);
        }
        if (idx >= contents.size())
            Log(LOG_FATAL) << "Exceed scope of bins";

        contents[idx]++;
    }

    return contents;
}

#endif   /* ----- #ifndef _COMMON_HISTOGRAM_H_  ----- */
