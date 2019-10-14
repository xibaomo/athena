/*
 * =====================================================================================
 *
 *       Filename:  bg_types.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/13/2019 12:33:17 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BOOST_GEOM_TYPES_H_
#define  _BOOST_GEOM_TYPES_H_

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/index/rtree.hpp>
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef float BGReal;
typedef bg::model::point<BGReal, 2, bg::cs::cartesian> bg_point_t;
typedef bg::model::box<bg_point_t> bg_box_t;
typedef bg::model::polygon<bg_point_t> bg_polygon_t;
typedef bg::model::segment<bg_point_t> bg_seg_t;
typedef std::pair<bg_box_t, unsigned int> bg_node_t;
typedef bgi::rtree<bg_node_t, bgi::quadratic<16>> bg_rtree_t;

#endif   /* ----- #ifndef _BOOST_GEOM_TYPES_H_  ----- */
