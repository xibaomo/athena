/*
 * =====================================================================================
 *
 *       Filename:  intersection.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/01/2019 07:29:56 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  intersection_INC
#define  intersection_INC

typedef double real_t;
void comp_seg_intersection(const real_t s1sx, const real_t s1sy, const real_t s1ex, const real_t s1ey,
                           const real_t s2sx, const real_t s2sy, const real_t s2ex, const real_t s2ey,
                           real_t& x, real_t&y);
#endif   /* ----- #ifndef intersection_INC  ----- */
