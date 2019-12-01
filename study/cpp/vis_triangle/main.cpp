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

    createPolys(pa);
    pa.appendPolygon(std::move(out));
    vis::TriEdgeContainer tec(&pa);
    tec.convertPolys();
    return 0;
}
