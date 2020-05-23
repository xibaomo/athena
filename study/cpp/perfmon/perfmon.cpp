/*
 * =====================================================================================
 *
 *       Filename:  perfmon.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  01/08/2020 08:05:19 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include <iostream>
#include <cmath>
#include "utils.h"
#define N 10000000
using namespace std;

int main() {
    MsTimer tm;
    for ( int i = 0; i < N; i++ ) {
//         i*0.0000001-i*0.000000002;
         sqrt((double)i);
    }
    cout << "It takes " << tm.elapsed() << endl;
    return 0;
}
