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
#include "utils.h"
using namespace std;
using namespace iht;

const int L = 100;

void createPolys(PolygonArray& pa) {
/*
    Polygon p;
    p.appendVertex(Point{20, 20});
    p.appendVertex(Point{80, 20});
    p.appendVertex(Point{80, 80});
    p.appendVertex(Point{20, 80});
    p.appendVertex(Point{20, 20});

    pa.appendPolygon(std::move(p));
    */

    for ( int i = 1; i < L-2; i+=96 ) {
        for ( int j = 1; j < L-2; j+=96 ) {
            Polygon p;
            p.appendVertex(Point{i, j});
            p.appendVertex(Point{i+1, j});
            p.appendVertex(Point{i+1, j+1});
            p.appendVertex(Point{i, j+1});
            p.appendVertex(Point{i, j});
            pa.appendPolygon(std::move(p));
        }
    }
    cout << "polygons: " << pa.size() << endl;
}
int main() {
    Polygon out;
    PolygonArray pa;
    out.appendVertex(Point{0, 0});
    out.appendVertex(Point{L, 0});

    out.appendVertex(Point{L, L});
    out.appendVertex(Point{0, L});
    out.appendVertex(Point{0, 0});

    pa.appendPolygon(std::move(out)); // boundary must be the 1st
    MsTimer tm;
    createPolys(pa);
    cout << "pre-proc takes " << tm.elapsed() << endl;
    vis::TriEdgeContainer tec(&pa);
    tec.convertPolys();

    vis::Edge entry_edge = std::make_pair(0, 1);
    const double vx = L*.5, vy = -1e-6;
    tec.setEntryEdge(entry_edge, vx, vy);

    tm.restart();
    vis::VisPoly vis_segs;
    tec.recurFindVisibleEdges(entry_edge, -1, vx, vy, M_PI, 0., vis_segs);
    cout << "vis takes " << tm.elapsed() << endl;

    cout << "visibility polygon: " << endl;
    for ( auto& it : vis_segs ) {
        cout << it.x << "," << it.y << ": " << it.polar_angle << endl;
    }
    return 0;
}
