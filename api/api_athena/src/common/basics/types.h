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

const Uint MAXPORTNUM = 65536;
const Uint MINPORTNUM = 1024;
#endif   /* ----- #ifndef _BASICS_TYPES_H_  ----- */
