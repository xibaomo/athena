/*
 * =====================================================================================
 *
 *       Filename:  tri.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  11/28/2019 02:08:13 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "earcut_inc.hpp"
#include <iostream>
#include "utils.h"
using namespace std;
using Point = std::array<double, 2>;
const int N = 1;
const double L = .5;

std::vector<std::vector<Point>> polygon;

int main() {
    polygon.push_back({{L, 0}, {L, L}, {0, L}, {0, 0}});
    for ( int i = 0; i < N; i++ ) {
        for ( int j = 0; j < N; j++ ) {
            polygon.push_back({{i+.1, j+.1}, {i+0.2, j+.1},
                               {i+.2, j+.2},
                               {i+.3, j+.2},
                               {i+0.3, j+.3}, {i+.1, j+.3}});
        }
    }

    cout << "num of polygons: " << polygon.size() << endl;
    MsTimer tm;

    std::vector<unsigned int> indices = mapbox::earcut<unsigned int>(polygon);
    cout << "tri takes " << tm.elapsed() << endl;

    cout << indices.size() << endl;
    for ( size_t v = 0; v < indices.size(); v+=3 ) {
        cout << indices[v] << "," << indices[v+1] << "," << indices[v+2] << endl;
    }

    return 0;
}
