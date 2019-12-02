/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/30/2019 01:23:46 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "iht_poly.hpp"
#include "vis_pt.h"
#include <iostream>
using namespace std;
using namespace iht;

//template class std::set<vis::PointAngle, vis::PointAngleComp>;

void createPolys(PolygonArray& pa) {
    Polygon p;
    p.appendVertex(Point{20, 20});
    p.appendVertex(Point{80, 20});
    p.appendVertex(Point{80, 80});
    p.appendVertex(Point{20, 80});
    p.appendVertex(Point{20, 20});

    pa.appendPolygon(std::move(p));
}
const int L = 100;
int main() {
    Polygon out;
    PolygonArray pa;
    out.appendVertex(Point{0, 0});
    out.appendVertex(Point{L, 0});
    out.appendVertex(Point{L, L});
    out.appendVertex(Point{0, L});
    out.appendVertex(Point{0, 0});

    pa.appendPolygon(std::move(out)); // boundary must be the 1st
    createPolys(pa);
    vis::TriEdgeContainer tec(&pa);
    tec.convertPolys();

    vis::Edge entry_edge = std::make_pair(0, 1);
    const double vx = L*.5, vy = -1e-6;
    tec.setEntryEdge(entry_edge, vx, vy);

    vis::VisPoly vis_segs;
    tec.recurFindVisibleEdges(entry_edge, -1, vx, vy, M_PI, 0., vis_segs);

    cout << "visibility polygon: " << endl;
    for ( auto& it : vis_segs ) {
        cout << it.x << "," << it.y << ": " << it.polar_angle << endl;
    }
    return 0;
}
