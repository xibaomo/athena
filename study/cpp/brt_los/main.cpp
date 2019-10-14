/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/11/2019 11:35:23 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "generic_poly.h"
#include "brt_container.h"
using namespace std;
int main(int argc, char** argv) {
    PolySet ps;
    ps.createPolys();

    BrtContainer* pc = &BrtContainer::getInstance();
    pc->loadPolys(&ps);
    const String layer = "245:101";
    pc->convertPolys("245:101");
    auto vs = pc->snapToEdge("245:101",-1, 3);
    cout << "(-1, 3) --> " << vs.x << " " << vs.y << endl;
    cout << vs.norm_dir << endl;

    vs = pc->snapToEdge("245:101",3, -1);
    cout << "(3, -1) --> " << vs.x << " " << vs.y << endl;
    cout << vs.norm_dir << endl;

    vs = pc->snapToEdge(layer, 6, 4);
    cout << "(6, 4) --> " << vs.x << " " << vs.y << endl;
    cout << vs.norm_dir << endl;

    vs = pc->snapToEdge(layer, 2, 12);
    cout << "(2, 12) --> " << vs.x << " " << vs.y << endl;
    cout << vs.norm_dir << endl;

    vs = pc->snapToEdge(layer, 3.5, 9.5);
    cout << "(3.5, 9.5) --> " << vs.x << " " << vs.y << endl;
    cout << vs.norm_dir << endl;

    return 0;
}
