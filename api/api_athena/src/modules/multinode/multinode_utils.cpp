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
using namespace std;
using namespace athena;
void pushEdges(String sym, real64 ask, real64 bid,
        vector<Edge>& G,
        set<String>& currencies)
{
    Edge a2b, b2a;
    String sym_x = sym.substr(0, 3);
    String sym_y = sym.substr(3, 3);
    currencies.insert(sym_x);
    currencies.insert(sym_y);
    a2b.a = sym_x;
    a2b.b = sym_y;
    b2a.a = sym_y;
    b2a.b = sym_x;

    a2b.w = -log(bid);
    b2a.w = -log(1./ask);

    G.push_back(a2b);
    G.push_back(b2a);
}
