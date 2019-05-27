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
    APP_PREDICTOR,
    APP_TICKCLASSIFIER,
};


// MinBar predictor
enum class MBP_Type : int {
    MA_PRED = 0,
    ML_EACH_MB
};
#endif   /* ----- #ifndef _BASICS_TYPES_H_  ----- */
