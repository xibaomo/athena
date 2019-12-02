/*
 * =====================================================================================
 *
 *       Filename:  intersection.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/01/2019 07:39:44 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "intersection.h"
#include <boost/geometry.hpp>
typedef boost::geometry::model::d2::point_xy<double> pt_t;
typedef boost::geometry::model::segment<pt_t> seg_t;
void comp_seg_intersection(const real_t s1sx, const real_t s1sy, const real_t s1ex, const real_t s1ey,
                           const real_t s2sx, const real_t s2sy, const real_t s2ex, const real_t s2ey,
                           real_t& x, real_t&y) {
    seg_t s1{{s1sx, s1sy}, {s1ex, s1ey}};
    seg_t s2{{s2sx, s2sy}, {s2ex, s2ey}};
    std::vector<pt_t> res;

    boost::geometry::intersection(s1, s2, res);

    x = boost::geometry::get<0>(res[0]);
    y = boost::geometry::get<1>(res[0]);
}

