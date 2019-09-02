/*
 * =====================================================================================
 *
 *       Filename:  multinode_utils.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/21/2019 11:20:28 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MULTINODE_UTILS_H_
#define  _MULTINODE_UTILS_H_
#include "basics/types.h"
#include <vector>
#include <vector>
struct Edge {
    int a, b;
    real64 w;
};

void pushEdges(String sym,
        real64 ask, real64 bid,
        //output
        std::vector<Edge>& G,
        std::vector<String>& currencies);

std::vector<int>
bellmanford(std::vector<Edge>& edges, int N);

void
addDummyNode(std::vector<Edge>& edges, std::vector<String>& nodes);
#endif   /* ----- #ifndef _MULTINODE_UTILS_H_  ----- */
