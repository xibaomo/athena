/*
 * =====================================================================================
 *
 *       Filename:  pair_labeler.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  07/04/2021 08:41:02 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */
#pragma once

#include "minbar_pair_trader/minbar_pair_trader.h"
class PairLabeler : public MinbarPairTrader {
protected:
    // 0 - buy take profit
    // 1 - buy stop loss
    // 2 - sell take profit
    // 3 - sell stop loss
    std::vector<int> m_pairLabels;

    PairLabeler(const String& cf) : MinbarPairTrader(cf) {
        Log(LOG_INFO) << "Pair labeler app created";
    }
public:
    ~PairLabeler();

    static PairLabeler& getInstance(const String& cf) {
        static PairLabeler _ins(cf);
        return _ins;
    }

    Message processMsg(Message& msg) override;
    Message procMsg_PAIR_MIN_OPEN(Message& msg) override;
    Message procMsg_PAIR_LABEL(Message& msg);
};
