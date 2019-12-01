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
namespace vis {
    typedef unsigned int uint;
    struct VertexInfo {
        iht::Point* host;
//        uint prev;
//        uint next;
    };

    struct EdgeInfo {
        std::set<uint> opp_pts; // pt id
        bool isOriginal;
    };

    class TriEdgeContainer {
        iht::PolygonArray* m_hostPolyArray;
        std::vector<VertexInfo> m_allVertices;
        std::map<std::pair<uint, uint>, EdgeInfo> m_edge2info;
   public:
        TriEdgeContainer(iht::PolygonArray* pa) : m_hostPolyArray(pa) {;}

        void convertPolys() {
            for ( auto& poly : *m_hostPolyArray ) {
                for ( auto it =poly.begin(); it!=poly.end(); it++ ) {
                    if ( std::next(it) == poly.end()) continue;
                    m_allVertices.push_back(VertexInfo{&(*it)});
                }
            }
            std::cout << m_allVertices.size() << std::endl;
        }
    };
}
#endif   /* ----- #ifndef vis_pt_INC  ----- */
