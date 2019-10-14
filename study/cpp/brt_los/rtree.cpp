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
#include <unordered_map>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/index/rtree.hpp>
using namespace std;
typedef std::string String;
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::d2::point_xy<int> point_t;
typedef bg::model::polygon<point_t> polygon_t;
typedef std::pair<polygon_t, unsigned int> node_t;

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
    PolySet ps;

    auto& polys = ps.getPolys("");

    bgi::rtree<node_t, bgi::quadratic<16>> tree;

    return 0;
}
