/*
 * =====================================================================================
 *
 *       Filename:  brt_container.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/12/2019 12:09:04 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BRT_CONTAINER_H_
#define  _BRT_CONTAINER_H_

#include "poly_container.h"
#include "bg_types.h"
#include "poly_profile.h"
class BrtContainer : public PolyContainer {
protected:
    std::unordered_map<String, std::vector<PolyProfile>> m_layer2polyProfiles;
    std::unordered_map<String, bg_rtree_t> m_layer2tree;
    BrtContainer() {;}
public:
    virtual ~BrtContainer() {;}

    static BrtContainer& getInstance() {
        static BrtContainer _ins;
        return _ins;
    }

    /**
     *  Convert polygons of the given layer to rtree
     */
    void convertPolys(const String& layer) {
        auto& polymap = m_polys->getPolyMap();
        auto& ps =  polymap[layer].polys;
        std::vector<PolyProfile> pps;
        for ( auto& itt : ps ) {
            int xmin, ymin, xmax, ymax;
            itt.getXYlimit(xmin, ymin, xmax, ymax);
            bg_box_t b(bg_point_t(xmin, ymin), bg_point_t(xmax, ymax));
            pps.emplace_back(PolyProfile(&itt, b));
        }

        m_layer2tree[layer] = bg_rtree_t();
        auto& rtree =m_layer2tree[layer];
        for ( size_t i = 0; i < pps.size(); i++ ) {
            rtree.insert(std::make_pair(pps[i].getBox(), i));
        }

        std::cout << "polygons inserted to rtree: " << layer << std::endl;
        m_layer2polyProfiles[layer] = std::move(pps);
    }

    /**
     * Snap to nearest polygon edge
     */
    ViewSite snapToEdge(const String& layer, BGReal x, BGReal y, BGReal rng = 5) {
        bg_box_t query_box(bg_point_t(x-rng, y-rng), bg_point_t(x+rng, y+rng));
        auto& rtree = m_layer2tree[layer];
        auto& pps   = m_layer2polyProfiles[layer];
        std::vector<bg_node_t> res;
        do {
            rng+=1.;
            res.clear();
            rtree.query(bgi::intersects(query_box), std::back_inserter(res));
        } while(res.empty() && rng < 10);

        if ( rng >=10) std::cout << "Cannot find host polygon" << std::endl;

        ViewSite vs; vs.snap_dist2 = 999999;
        for ( auto& r : res ) {
            ViewSite vst = findNearestSpotOnPolygon(x, y, pps[r.second]);
            if ( vst.snap_dist2 < vs.snap_dist2 )
                vs = vst;
        }

        return vs;
    }

    ViewSite findNearestSpotOnPolygon(BGReal x, BGReal y, PolyProfile& pp) {
         auto& pts = pp.getHostPoly()->pts;
         ViewSite vs;
         vs.snap_dist2 = 1e10;
         for ( auto its = pts.begin(); its!=pts.end(); its++ ) {
             auto ite = std::next(its);
             if ( ite == pts.end()) break;
             bg_point_t ps = bg_point_t(its->x, its->y);
             bg_point_t pe = bg_point_t(ite->x, ite->y);
             bg_point_t pg = bg_point_t(x, y);
             bg_point_t ve = bg_point_t(ite->x-its->x, ite->y-its->y);
             bg_point_t vg = bg_point_t(x-its->x, y-its->y);
             BGReal edge_len = bg::distance(ps, pe);
             BGReal proj = bg::dot_product(vg, ve)/edge_len;
             BGReal dist2;
             bg_point_t snp ; //snapped target
             if ( proj<=0 ) {
                 dist2 = bg::comparable_distance(pg, ps);
                 snp = ps;
             } else if (proj >= edge_len) {
                 dist2 = bg::comparable_distance(pg, pe);
                 snp = pe;
             } else {
                 dist2 = bg::comparable_distance(pg, ps) - proj*proj;
                 BGReal xt = bg::get<0>(ps) + bg::get<0>(ve)/edge_len * proj;
                 BGReal yt = bg::get<1>(ps) + bg::get<1>(ve)/edge_len * proj;
                 bg::set<0>(snp, xt);
                 bg::set<1>(snp, yt);
             }
             if ( dist2 < vs.snap_dist2 ) {
                 vs.x = bg::get<0>(snp);
                 vs.y = bg::get<1>(snp);
                 vs.snap_dist2 = dist2;
                 std::complex<BGReal> ce(bg::get<0>(ve)/edge_len, bg::get<1>(ve)/edge_len);
                 vs.norm_dir = rot90cw(ce);
             }
        }

        return vs;
    }
    std::complex<BGReal> rot90cw(std::complex<BGReal>& c) {
        return c * std::complex<BGReal>(0, -1);
    }
};
#endif   /* ----- #ifndef _BRT_CONTAINER_H_  ----- */
