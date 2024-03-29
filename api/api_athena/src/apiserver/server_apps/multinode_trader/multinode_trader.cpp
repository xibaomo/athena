/*
 * =====================================================================================
 *
 *       Filename:  multinode_trader.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/24/2019 01:09:21 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "multinode_trader.h"
#include "multinode/multinode_utils.h"
#include "basics/utils.h"

using namespace std;
using namespace athena;

Message
MultinodeTrader::processMsg(Message& msg)
{
    Message outmsg;
    FXAct act = (FXAct)msg.getAction();

    switch(act) {
    case FXAct::ALL_SYM_OPEN:
        outmsg = procMsg_ALL_SYM_OPEN(msg);
        break;
    default:
        break;
    }

    return outmsg;
}
Message
MultinodeTrader::procMsg_ALL_SYM_OPEN(Message& msg)
{
    real64* pm = (real64*)msg.getData();
    String cmt = msg.getComment();
    vector<String> syms = splitString(cmt,",");
    syms.pop_back();

    m_symGraph.clear();
    m_currencies.clear();

    for(size_t i=0;i<syms.size();i++) {
        pushEdges(syms[i],pm[2*i],pm[2*i+1],m_symGraph,m_currencies);
    }

    addDummyNode(m_symGraph,m_currencies);

    Log(LOG_INFO) << "Received currencies: " + to_string(m_currencies.size()) <<std::endl;

    vector<int> cycle = bellmanford(m_symGraph,m_currencies.size());
    if (cycle.empty()) {
        //Log(LOG_INFO) << "No negative-weight cycle found" <<std::endl;
    } else {
        Log(LOG_INFO) << "Arbitrage loop found:" <<std::endl;
        String loop;
        for(auto c: cycle) {
            loop+=m_currencies[c] + "-->";
        }
        loop.erase(loop.size()-3,3);
        Log(LOG_INFO) << loop <<std::endl;
    }

    Message out;
    out.setAction(FXAct::NOACTION);
    return out;

}
