/*
 * =====================================================================================
 *
 *       Filename:  generic_poly.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/11/2019 11:30:29 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _GENERIC_POLYGON_TYPE_H_
#define  _GENERIC_POLYGON_TYPE_H_

#include <list>
#include <unordered_map>
#include <cstring>
#include <vector>
typedef std::string String;
struct Point {
    int x, y;
};

typedef std::list<Point> PointGroup;
struct Polygon{
    PointGroup pts;
    void getXYlimit(int& xmin, int& ymin, int& xmax, int& ymax) {
        xmin = xmax = pts.begin()->x;
        ymin = ymax = pts.begin()->y;
        for ( auto& p : pts ) {
            if ( xmin > p.x ) xmin = p.x;
            if ( xmax < p.x) xmax = p.x;
            if ( ymin > p.y) ymin = p.y;
            if ( ymax < p.y) ymax = p.y;
        }
    }
};

struct PolyArray {
    std::vector<Polygon> polys;
};
class PolySet {
    std::unordered_map<String, PolyArray> m_layer2polys;
public:
    std::unordered_map<String, PolyArray>& getPolyMap() { return m_layer2polys; }
    PolyArray& getPolys(const String& layer) { return m_layer2polys[layer]; }

    void createPolys() {
        const int w = 5, h = 10, gap = 20;
        //poly1
        Polygon p1;
        p1.pts.emplace_back(Point{0, 0});
        p1.pts.emplace_back(Point{w, 0});
        p1.pts.emplace_back(Point{w, h});
        p1.pts.emplace_back(Point{0, h});
        p1.pts.emplace_back(Point{0, 0});
        //poly2
        Polygon p2;
        p2.pts.emplace_back(Point{0-gap-w, 0});
        p2.pts.emplace_back(Point{0-gap, 0});
        p2.pts.emplace_back(Point{0-gap, h});
        p2.pts.emplace_back(Point{0-gap-w, h});
        p2.pts.emplace_back(Point{0-gap-w, 0});
        //poly3
        Polygon p3;
        p3.pts.emplace_back(Point{0, h+gap});
        p3.pts.emplace_back(Point{w, h+gap});
        p3.pts.emplace_back(Point{w, h+gap+h});
        p3.pts.emplace_back(Point{0, h+gap+h});
        p3.pts.emplace_back(Point{0, h+gap});
        //poly4
        Polygon p4;
        p4.pts.emplace_back(Point{w+gap, 0});
        p4.pts.emplace_back(Point{w+gap+w, 0});
        p4.pts.emplace_back(Point{w+gap+w, h});
        p4.pts.emplace_back(Point{w+gap, h});
        p4.pts.emplace_back(Point{w+gap, 0});

        PolyArray pa;
        pa.polys.emplace_back(p1);
        pa.polys.emplace_back(p2);
        pa.polys.emplace_back(p3);
        pa.polys.emplace_back(p4);

        m_layer2polys["245:101"] = pa;
        m_layer2polys["51:0"] = pa;
    }
};
#endif   /* ----- #ifndef _GENERIC_POLYGON_TYPE_H_  ----- */
