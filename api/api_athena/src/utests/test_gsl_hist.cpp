/*
 * =====================================================================================
 *
 *       Filename:  test_gsl_hist.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  07/25/2019 01:54:17 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "basics/csv_parser.h"
#include "basics/types.h"
#include <vector>
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
    io::CSVReader<3> in(argv[1]);
    in.read_header(io::ignore_extra_column,"x","y","spread");
    real32 x,y,spread;

    vector<real32> xs,ys,sps;
    while(in.read_row(x,y,spread)) {
        xs.push_back(x);
        ys.push_back(y);
        sps.push_back(spread);
    }

    for (auto& v : sps) {
        cout<<v<<endl;
    }

    return 0;
}
