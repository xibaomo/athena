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
#include "gsl/gsl_histogram.h"
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

    gsl_histogram_pdf* hp = gsl_histogram_pdf_alloc(100);
    gsl_histogram *h = gsl_histogram_alloc(100);
    gsl_histogram_set_ranges_uniform(h,-5,5);
    for(auto v: sps) {
        gsl_histogram_increment(h,v);
    }

    gsl_histogram_pdf_init(hp,h);

    gsl_histogram_pdf_free(hp);
    gsl_histogram_free(h);

    return 0;
}
