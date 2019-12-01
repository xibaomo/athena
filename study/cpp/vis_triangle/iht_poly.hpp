/*
 * =====================================================================================
 *
 *       Filename:  iht_poly.hpp
 *
 *    Description:  mimic iht polygons
 *
 *        Version:  1.0
 *        Created:  11/30/2019 01:16:23 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef  iht_poly_INC
#define  iht_poly_INC
#include <list>
#include <utility>
namespace iht {
    struct Point {
        int x,y;
    };


    /*-----------------------------------------------------------------------------
     *  Polygon
     *-----------------------------------------------------------------------------*/
    typedef std::list<Point> PointGroup;
    class Polygon {
        PointGroup m_vertices;
   public:
        typedef PointGroup::iterator iterator;
        void appendVertex(Point&& pt) {
            m_vertices.push_back(std::forward<Point>(pt));
        }
        std::size_t size() { return m_vertices.size();}
        iterator begin() { return m_vertices.begin();}
        iterator end() { return m_vertices.end();}
    };


    /*-----------------------------------------------------------------------------
     *  Polygon array
     *-----------------------------------------------------------------------------*/
    class PolygonArray {
        std::list<Polygon> m_polys;
   public:
//        typedef std::list<Polygon>::iterator iterator;
        using iterator=std::list<Polygon>::iterator;
        void appendPolygon(Polygon&& poly) {
            m_polys.push_back(std::forward<Polygon>(poly));
        }
        std::size_t size() { return m_polys.size(); }
        iterator begin() { return m_polys.begin(); }
        iterator end() { return m_polys.end();}
    };
}
#endif   /* ----- #ifndef iht_poly_INC  ----- */
