/*
 * =====================================================================================
 *
 *       Filename:  rtree.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/11/2019 12:27:55 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#include <list>
#include <iostream>
#include <unordered_map>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/index/rtree.hpp>
#include "utils.h"
using namespace std;
typedef std::string String;
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::d2::point_xy<double> point_t;
typedef bg::model::box<point_t> box_t;
typedef bg::model::polygon<point_t> polygon_t;
typedef bg::model::segment<point_t> segment_t;
typedef std::pair<segment_t, unsigned int> node_t;

struct Point {
    int x, y;
};
typedef std::list<Point> Polygon;
class PolySet {
private:
    unordered_map<String, vector<Polygon>> m_layer2Polys;
public:
    vector<Polygon>& getPolys(const String& s) { return m_layer2Polys[s]; }
};

int main() {
    const int N = 50;
    std::list<node_t> segs;

    int k = 0;
    for ( int i = 0; i < N; i++ ) {
        for ( int j = 0; j < N; j++ ) {
            segment_t s(point_t(i, j), point_t(i+.5, j+.5));
            segs.emplace_back(std::make_pair(s, k++));
        }
    }
    MsTimer tim;
    bgi::rtree<node_t, bgi::quadratic<64, 2>> tree(segs);
    cout << "build takes " << tim.elapsed() << endl;

    tim.restart();
    box_t b(point_t(3, 3), point_t(20, 30));
    std::vector<node_t> res; res.reserve(1004);
    tree.query(bgi::intersects(b), std::back_inserter(res));
//    for ( auto it = tree.qbegin(bgi::intersects(b)); it!=tree.qend(); it++ ) {
//        res.push_back(*it);
//    }
    cout << "query takes " << tim.elapsed() << endl;
    cout << "segs : " << res.size() << endl;
    return 0;
}
