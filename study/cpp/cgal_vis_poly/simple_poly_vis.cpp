/*
 * =====================================================================================
 *
 *       Filename:  simple_poly_vis.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/01/2019 01:19:18 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Simple_polygon_visibility_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arr_naive_point_location.h>
#include <istream>
#include <vector>
#include "utils.h"
using namespace std;

typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Segment_2 Segment_2;
typedef CGAL::Arr_segment_traits_2<Kernel> Traits_2;
typedef CGAL::Arrangement_2<Traits_2> Arrangement_2;
typedef Arrangement_2::Face_handle  Face_handle;
typedef Arrangement_2::Edge_const_iterator Edge_const_iterator;
typedef Arrangement_2::Ccb_halfedge_circulator Ccb_halfedge_circulator;

int main() {
    // create environment
    Point_2 p1(0, 4), p2(0, 0), p3(3, 2), p4(4, 0), p5(4, 4), p6(1, 2);
    std::vector<Segment_2> segments;
    segments.push_back(Segment_2(p1, p2));
    segments.push_back(Segment_2(p2, p3));
    segments.push_back(Segment_2(p3, p4));
    segments.push_back(Segment_2(p4, p5));
    segments.push_back(Segment_2(p5, p6));
    segments.push_back(Segment_2(p6, p1));

    MsTimer tim;
    Arrangement_2 env;
    CGAL::insert_non_intersecting_curves(env, segments.begin(), segments.end());
    // find the face of the query point
    Point_2 q(0.000000001, 2);
    Arrangement_2::Face_const_handle *face;
    CGAL::Arr_naive_point_location<Arrangement_2> pl(env);
    CGAL::Arr_point_location_result<Arrangement_2>::Type obj = pl.locate(q);
    // the query point locates in the interior of a face
    face = boost::get<Arrangement_2::Face_const_handle> (&obj);

    // compute non regularized visibility area
    // define visibility object type that computes non-regularized visibility area
//    typedef CGAL::Simple_polygon_visibility_2<Arrangement_2, CGAL::Tag_false> NSPV;
    typedef CGAL::Simple_polygon_visibility_2<Arrangement_2, CGAL::Tag_true> NSPV;
    Arrangement_2 non_regular_output;
    NSPV non_regular_visibility(env);
    non_regular_visibility.compute_visibility(q, *face, non_regular_output);
    cout << "vis poly takes " << tim.elapsed() << endl;

    cout << "visibility area of q has edges: " << non_regular_output.number_of_edges() << endl;
    for ( Edge_const_iterator it = non_regular_output.edges_begin(); it!=non_regular_output.edges_end(); ++it ) {
        cout << "[" << it->source()->point() << " -> " << it->target()->point() << "]" << endl;
    }
    cout << sizeof(Point_2) << endl;
    return 0;
}
