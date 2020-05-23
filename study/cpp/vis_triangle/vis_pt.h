/*
 * =====================================================================================
 *
 *       Filename:  vis_pt.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/30/2019 05:39:46 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  vis_pt_INC
#define  vis_pt_INC
#include "iht_poly.hpp"
#include <set>
#include <unordered_map>
#include <map>
#include <iostream>
#include <vector>
#include <complex>
#include <math.h>
#include "earcut_inc.hpp"
#include "intersection.h"
namespace vis {
    typedef unsigned int uint;
    typedef std::array<double, 2> Point;
    typedef std::array<Point, 2>  Segment;
    typedef std::pair<uint, uint> Edge;

    double Arg(std::complex<double>&& cmplx) {
        double ag = std::arg(cmplx);
        if ( ag < 0) ag += M_PI*2.;
        return ag;
    }
    double Arg(std::complex<double>& cmplx) {
        double ag = std::arg(cmplx);
        if ( ag < 0) ag += M_PI*2.;
        return ag;
    }
    struct VertexInfo {
        iht::Point* host;
        int prev;
        int next;
    };

    struct VertexAngle {
        uint vtx_id;
        double polar_angle;
    };
    struct PointAngle {
        double x, y;
        double polar_angle;
    };
    struct PointAngleComp {
        bool operator()(const PointAngle& p1, const PointAngle& p2) {
            return p1.polar_angle < p2.polar_angle;
        }
    };
    typedef std::multiset<PointAngle, PointAngleComp> VisPoly;
    struct EdgeInfo {
        std::set<uint> opp_pts; // pt id
        bool isTrueEdge;
        std::pair<VertexAngle, VertexAngle> vtx_angles; //sorted
    };

    class TriEdgeContainer {
        iht::PolygonArray* m_hostPolyArray;
        std::vector<VertexInfo> m_allVertices;
        std::map<Edge, EdgeInfo> m_edge2info;
        double m_losRange;
   public:
        TriEdgeContainer(iht::PolygonArray* pa) : m_hostPolyArray(pa) {m_losRange = 1000/0.05; }

        /*-----------------------------------------------------------------------------
         *  check if a segment is true
         *-----------------------------------------------------------------------------*/
        bool checkTrueEdge(const std::pair<uint, uint>& s) {
            auto& vt = m_allVertices[s.first];
            if ( s.second == vt.prev || s.second == vt.next ) {
                return true;
            }
            return false;
        }

        /*-----------------------------------------------------------------------------
         *  Add edges of a triangle to m_edge2info
         *-----------------------------------------------------------------------------*/
        void procATriangle(uint i, uint j, uint k) {
            auto s1 = std::minmax(i, j);
            m_edge2info[s1].opp_pts.insert(k);
            m_edge2info[s1].isTrueEdge = checkTrueEdge(s1);
            auto s2 = std::minmax(i, k);
            m_edge2info[s2].opp_pts.insert(j);
            m_edge2info[s2].isTrueEdge = checkTrueEdge(s2);
            auto s3 = std::minmax(k, j);
            m_edge2info[s3].opp_pts.insert(i);
            m_edge2info[s3].isTrueEdge = checkTrueEdge(s3);
        }

        /*-----------------------------------------------------------------------------
         *  Add edges of all triangles to m_edge2info
         *-----------------------------------------------------------------------------*/
        void procAllTriangles(std::vector<uint>& indices) {
            for ( size_t i = 0; i < indices.size(); i+=3 ) {
                procATriangle(indices[i], indices[i+1], indices[i+2]);
            }
        }

        /*-----------------------------------------------------------------------------
         *  Convert polygons to triangles and save info of all edges
         *-----------------------------------------------------------------------------*/
        void convertPolys() {
            std::vector<std::vector<Point>> tmp_polys;
            int pt_id(0);
            for ( auto& poly : *m_hostPolyArray ) {
                std::vector<Point> tmppoly;
                uint head_id = pt_id;
                for ( auto it =poly.begin(); it!=poly.end(); it++ ) {
                    if ( std::next(it) == poly.end() ) {
                        m_allVertices[head_id].prev = pt_id-1;
                        m_allVertices[pt_id-1].next = head_id;
                        continue;
                    }
                    tmppoly.emplace_back(Point{it->x*1., it->y*1.});
                    m_allVertices.push_back(VertexInfo{&(*it), pt_id-1, pt_id+1});
                    pt_id++;
                }
                tmp_polys.emplace_back(std::move(tmppoly));
            }
//            std::cout << m_allVertices.size() << std::endl;
            std::vector<uint> inds = mapbox::earcut(tmp_polys);
//            std::cout << inds.size() << std::endl;
            for ( size_t i =0; i < inds.size(); i+=3 ) {
                std::cout << inds[i] << "," << inds[i+1] << "," << inds[i+2] << std::endl;
            }
            procAllTriangles(inds);
            for ( auto& it : m_edge2info ) {
//                std::cout << it.first.first << "-" << it.first.second << ": " << it.second.isTrueEdge << std::endl;
            }
        }

        /*-----------------------------------------------------------------------------
         *  Set entry edge, whose view angles are 0 & pi
         *-----------------------------------------------------------------------------*/

        void setEntryEdge(const Edge& entry_edge, const double vx, const double vy) {
            if ( !isEdgeVisible(vx, vy, entry_edge, 2*M_PI, -1.e-5) ) {
                //log fatal
            }
        }
        int findOtherOppPtOfEdge(const std::pair<uint, uint>& edge, const int opp_pt) {
            auto& info = m_edge2info[edge];
            if ( opp_pt<0) return *(info.opp_pts.begin());
            for ( auto& it : info.opp_pts ) {
                if ( it != opp_pt) return it;
            }
            return -1;
        }
        bool isEdgeVisible(const  double vx, const double vy,
                           const Edge& e, const double lva, const double rva) {
            std::complex<double> c1(m_allVertices[e.first].host->x-vx, m_allVertices[e.first].host->y-vy);
            std::complex<double> c2(m_allVertices[e.second].host->x-vx, m_allVertices[e.second].host->y-vy);
            m_edge2info[e].vtx_angles = std::minmax(VertexAngle{e.first, Arg(c1)},
                                                    VertexAngle{e.second, Arg(c2)},
                                                    [](const VertexAngle& v1, const VertexAngle& v2) {
                                                    return v1.polar_angle < v2.polar_angle;
                                                    });

            auto& args = m_edge2info[e].vtx_angles;
            if ( args.first.polar_angle >= lva || args.second.polar_angle <= rva )
                return false;

            return true;
        }
        void extractVisibleSegment(const Edge& e, const double vx, const double vy, const double lva, const double rva,
                                   VisPoly& vis_seg) {
            auto& args = m_edge2info[e].vtx_angles;

            double clx, cly, crx, cry;
            if ( args.first.polar_angle < rva && args.second.polar_angle > lva ) {
                // 2 intersections
                comp_seg_intersection(m_allVertices[e.first].host->x, m_allVertices[e.first].host->y,
                        m_allVertices[e.second].host->x, m_allVertices[e.second].host->y,
                        vx, vy, vx+cos(lva)*m_losRange, vy+sin(lva)*m_losRange, clx, cly);
                comp_seg_intersection(m_allVertices[e.first].host->x, m_allVertices[e.first].host->y,
                        m_allVertices[e.second].host->x, m_allVertices[e.second].host->y,
                        vx, vy, vx+cos(rva)*m_losRange, vy+sin(rva)*m_losRange, crx, cry);
            }
            else if (args.first.polar_angle >=  rva && args.second.polar_angle > lva) {
                // left end outside, right end inside
                comp_seg_intersection(m_allVertices[e.first].host->x, m_allVertices[e.first].host->y,
                        m_allVertices[e.second].host->x, m_allVertices[e.second].host->y,
                        vx, vy, vx+cos(lva)*m_losRange, vy+sin(lva)*m_losRange, clx, cly);
                auto id = args.first.vtx_id;
                crx = m_allVertices[id].host->x;
                cry = m_allVertices[id].host->y;
            }
            else if (args.first.polar_angle < rva && args.second.polar_angle <= lva) {
                // right end outside, left inside
                comp_seg_intersection(m_allVertices[e.first].host->x, m_allVertices[e.first].host->y,
                        m_allVertices[e.second].host->x, m_allVertices[e.second].host->y,
                        vx, vy, vx+cos(rva)*m_losRange, vy+sin(rva)*m_losRange, crx, cry);
                auto id = args.second.vtx_id;
                clx = m_allVertices[id].host->x;
                cly = m_allVertices[id].host->y;
            } else {
                // both ends inside
                clx = m_allVertices[args.second.vtx_id].host->x;
                cly = m_allVertices[args.second.vtx_id].host->y;
                crx = m_allVertices[args.first.vtx_id].host->x;
                cry = m_allVertices[args.first.vtx_id].host->y;
            }

            auto argl = Arg(std::complex<double>(clx-vx, cly-vy));
            vis_seg.insert(PointAngle{clx, cly, argl});
            auto argr = Arg(std::complex<double>(crx-vx, cry-vy));
            vis_seg.insert(PointAngle{crx, cry, argr});
        }
        /*-----------------------------------------------------------------------------
         *  Find visible edges from a edge
         *-----------------------------------------------------------------------------*/
        void recurFindVisibleEdges(const std::pair<uint, uint>& entry_edge,
                                   const int opp_pt_outside,
                                   const double vx, const double vy, // view point
                                   double lva, double rva, // left right polar angles
                                   VisPoly& vis_segs) {
            // adjust view angle spanned by entry_edge
            auto& vas = m_edge2info[entry_edge].vtx_angles;
            if ( vas.first.polar_angle > rva) rva = vas.first.polar_angle;
            if ( vas.second.polar_angle < lva) lva = vas.second.polar_angle;

            int opp_pt_inside = findOtherOppPtOfEdge(entry_edge, opp_pt_outside);

            if ( opp_pt_inside< 0 ) return; //log fatal
            Edge e1 = std::minmax(entry_edge.first, (uint)opp_pt_inside);
            Edge e2 = std::minmax(entry_edge.second, (uint)opp_pt_inside);
            bool isE1Seen = isEdgeVisible(vx, vy, e1, lva, rva);
            bool isE2Seen = isEdgeVisible(vx, vy, e2, lva, rva);
            bool isE1True = m_edge2info[e1].isTrueEdge;
            bool isE2True = m_edge2info[e2].isTrueEdge;
            if ( !isE1Seen && !isE2Seen) return;
            if ( isE1Seen && isE1True ) {
                extractVisibleSegment(e1, vx, vy, lva, rva, vis_segs);
           }
            if ( isE2Seen && isE2True ) {
                extractVisibleSegment(e2, vx, vy, lva, rva, vis_segs);
            }
            if ( isE1Seen && isE1True && isE2Seen && isE2True) return; //dead end

            // if E1 is transparent, recursive
            if ( isE1Seen && !isE1True ) {
               recurFindVisibleEdges(e1, entry_edge.second, vx, vy, lva, rva, vis_segs);
            }

            // if E2 is transparent
            if ( isE2Seen && !isE2True ) {
               recurFindVisibleEdges(e2, entry_edge.first, vx, vy, lva, rva, vis_segs);
            }
        }
    };
}
#endif   /* ----- #ifndef vis_pt_INC  ----- */
