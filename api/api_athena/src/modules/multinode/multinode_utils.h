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
#include <set>
struct Edge {
    String a, b;
    real64 w;
};

void pushEdges(String sym,
        real64 ask, real64 bid,
        std::vector<Edge>& G,
        std::set<String>& currencies);
#endif   /* ----- #ifndef _MULTINODE_UTILS_H_  ----- */
