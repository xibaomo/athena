/*
 * =====================================================================================
 *
 *       Filename:  types.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 12:48:35 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BASICS_TYPES_H_
#define  _BASICS_TYPES_H_
#include <string>

#define ONE_MS 1
#define ONE_HUNDRED_MS 100

typedef unsigned int Uint;
typedef unsigned char Uchar;
using String = std::string;
typedef float real32;
typedef double real64;
typedef int int32;

#ifdef __USE_64_BIT
typedef double Real;
#define REALFORMAT "d"
#else
#define __USE_32_BIT
typedef float Real;
#define REALFORMAT "f"
#endif

enum AppType {
    APP_MINBAR_TRACKER = 0,
    APP_MINBARCLASSIFIER,
    APP_MINBAR_PAIRTRADER,
    APP_PAIR_SELECTOR,
    APP_ROBUST_PAIRTRADER, // 4
    APP_MULTINODE_TRADER = 5,
    APP_PREDICTOR,
    APP_TICKCLASSIFIER,
    APP_PAIR_LABELER, // 8
};

// MinBar predictor
enum class MBP_Type : int {
    MA_PRED = 0,
    ML_EACH_MB
};

#define NUM_MINBAR_FIELDS 6
struct MinBar {
    String time;
    real32 open;
    real32 high;
    real32  low;
    real32 close;
    int32  tickvol;

    MinBar(String p_time,real32 p_open,real32 p_high,real32 p_low,real32 p_close,int32 p_tickvol):
        time(std::move(p_time)),open(p_open),high(p_high),low(p_low),close(p_close),tickvol(p_tickvol)
        {
            ;
        }
};
#endif   /* ----- #ifndef _BASICS_TYPES_H_  ----- */
