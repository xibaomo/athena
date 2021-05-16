/*
 * =====================================================================================
 *
 *       Filename:  multinode_trader.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/24/2019 01:08:39 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _MULTINODE_TRADER_H_
#define  _MULTINODE_TRADER_H_
#include "server_apps/server_base_app/server_base_app.h"
#include "multinode/multinode_utils.h"
#include <vector>
/**
 *  Find a profitable loop in all the given sym pairs.
 */

class MultinodeTrader : public ServerBaseApp {
protected:
    std::vector<Edge> m_symGraph;
    std::vector<String> m_currencies;

    MultinodeTrader(const String& cf) : ServerBaseApp(cf) {
    }
public:
    virtual ~MultinodeTrader() {;}

    static MultinodeTrader& getInstance(const String& cf) {
        static MultinodeTrader _ins(cf);
        return _ins;
    }

    void prepare() {;}

    Message processMsg(Message& msg);

    Message procMsg_ALL_SYM_OPEN(Message& msg);

};
#endif   /* ----- #ifndef _MULTINODE_TRADER_H_  ----- */
