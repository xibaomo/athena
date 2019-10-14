/*
 * =====================================================================================
 *
 *       Filename:  poly_profile.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/13/2019 12:29:17 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _POLY_PROFILE_H_
#define  _POLY_PROFILE_H_

#include "generic_poly.h"
#include "bg_types.h"
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <complex>
struct ViewSite{
    BGReal x, y;
    std::complex<BGReal> norm_dir;
    BGReal snap_dist2;
};
class PolyProfile {
    Polygon* m_hostPoly;
    bg_box_t m_box;
    boost::shared_ptr<bg_polygon_t> m_auxPoly;
public:
    PolyProfile(Polygon* poly, bg_box_t& b) : m_hostPoly(poly), m_box(b){
    }

    Polygon* getHostPoly() { return m_hostPoly; }

    bg_box_t& getBox() { return m_box; }

    boost::shared_ptr<bg_polygon_t> getAuxPoly() {
        if ( !m_auxPoly ) genAuxPoly();
        return m_auxPoly;
    }

    /**
     * Generate aux polygon
     */
    void genAuxPoly() {
        m_auxPoly = boost::shared_ptr<bg_polygon_t>(new bg_polygon_t());
        for ( auto& it : m_hostPoly->pts ) {
            m_auxPoly->outer().emplace_back(bg_point_t(it.x, it.y));
        }
    }
};
#endif   /* ----- #ifndef _POLY_PROFILE_H_  ----- */
