/*
 * =====================================================================================
 *
 *       Filename:  multinode_utils.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/21/2019 11:35:50 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "multinode_utils.h"
#include "basics/utils.h"
#include "basics/log.h"
#include <cmath>
using namespace std;
using namespace athena;

void pushEdges(String sym, real64 ask, real64 bid,
               vector<Edge>& G,
               vector<String>& curs) {
    Edge a2b, b2a;
    String sym_x = sym.substr(0, 3);
    String sym_y = sym.substr(3, 3);

    int kx = searchVector(curs,sym_x);
    if (kx < 0) {
        curs.push_back(sym_x);
        kx =  curs.size()-1;
    }
    int ky = searchVector(curs,sym_y);
    if (ky < 0) {
        curs.push_back(sym_y);
        ky = curs.size()-1;
    }

    a2b.a = kx;
    a2b.b = ky;
    b2a.a = ky;
    b2a.b = kx;

    a2b.w = -log(bid);
    b2a.w = -log(1./ask);

    G.push_back(a2b);
    G.push_back(b2a);
}

vector<int> bellmanford(vector<Edge>& edges, int N) {
    vector<real64> d(N, DBL_MAX);
    vector<int> p(N, -1);
    d[N-1]=0;
    int x;
    vector<int> cycle;
    for ( int i = 0; i < N; i++ ) {
        x = -1;
        for ( Edge e: edges ) {
            if ( d[e.a] + e.w < d[e.b] ) {
                d[e.b] = d[e.a] + e.w;
                p[e.b] = e.a;
                x = e.b;
            }
        }
    }
    if ( x == -1 ) {
        Log(LOG_INFO) << "No negative cycle found";

    } else {
        for ( int i = 0; i < N; i++ )
            x = p[x];

        for ( int v = x; ;   v = p[v] ) {
            cycle.push_back(v);
            if ( v == x && cycle.size()>1 )
                break;
        }
        std::reverse(cycle.begin(), cycle.end());

//        cout << "Negative cycle: " << endl;
//        for ( int v: cycle )
//            cout << v << " ";
//        cout << endl;
    }

    return cycle;
}

void
addDummyNode(vector<Edge>& edges, vector<String>& nodes)
{
    nodes.push_back("XXX");
    int s = nodes.size()-1;
    for(size_t i=0; i < nodes.size(); i++) {
        Edge e;
        e.a = s;
        e.b = i;
        e.w = 0;
        edges.push_back(e);
    }
}


