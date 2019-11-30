#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Projection_traits_xy_3<K>  Gt;
typedef CGAL::Delaunay_triangulation_2<Gt> Delaunay;

typedef K::Point_3   Point;
const double L = 1000;

int main()
{
    std::vector<Point> poly;
    poly.emplace_back(0, 0, 1);
    poly.emplace_back(L, 0, 1);
    poly.emplace_back(L, L, 1);
    poly.emplace_back(0, L, 1);
    poly.emplace_back(0, 0, 1);

  Delaunay dt(poly.begin(), poly.end());
  std::cout << dt.number_of_vertices() << std::endl;
  return 0;
}
