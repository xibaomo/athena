/*
 * =====================================================================================
 *
 *       Filename:  pair_labeler.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  07/04/2021 08:48:25 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "pair_labeler.h"
#include "basics/utils.h"
#include "mean_revert/mean_revert.h"
using namespace std;
using namespace athena;

PairLabeler::~PairLabeler() {
    MeanRevert* oracle = dynamic_cast<MeanRevert*>(m_oracle);
    if (!oracle) Log(LOG_FATAL) << "not MeanRevert";

    auto& ts = oracle->getTradeSpreads();
    std::vector<real64> bs,ss;
    for(auto& v : ts) {
        bs.push_back(v.buy);
        ss.push_back(v.sell);
    }

    dumpVectors("labels.csv",bs,ss,m_pairLabels);
}
Message
PairLabeler::processMsg(Message& msg) {
    FXAct act = (FXAct)msg.getAction();
    Message outmsg;

    switch(act) {
    case FXAct::PAIR_LABEL:
        outmsg = procMsg_PAIR_LABEL(msg);
        break;
    case FXAct::PAIR_MIN_OPEN:
        outmsg = procMsg_PAIR_MIN_OPEN(msg);
        break;
    default:
        outmsg = MinbarPairTrader::processMsg(msg);
        break;
    }

    return outmsg;
}

Message
PairLabeler::procMsg_PAIR_LABEL(Message& msg) {
    int* pm = (int*)msg.getData();
    m_pairLabels[pm[0]] = pm[1]; // first ID, second label

    Message outmsg;
    return outmsg;
}

Message
PairLabeler::procMsg_PAIR_MIN_OPEN(Message& msg) {
    Message bm = MinbarPairTrader::procMsg_PAIR_MIN_OPEN(msg);
    bm.setAction(FXAct::NOACTION);
    m_pairLabels.push_back(-1);
    return bm;
}
