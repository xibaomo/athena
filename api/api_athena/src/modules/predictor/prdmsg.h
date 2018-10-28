/*
 * =====================================================================================
 *
 *       Filename:  prdmsg.h
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 08:53:20 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _PREDICTOR_PRDMSG_H_
#define  _PREDICTOR_PRDMSG_H_

#include "messenger/msg.h"

class PrdMessage : public Message {
public:
    enum class Action {
        GET_READY = 10,
        PREDICT
    };

    PrdMessage(Action action = Action::GET_READY,
            const size_t dataBytes = 0, const size_t charBytes = 0) : Message(dataBytes, charBytes) {
        setAction((ActionType)Action::GET_READY);
    }
};

typedef PrdMessage::Action PrdAction;
#endif   /* ----- #ifndef _PREDICTOR_PRDMSG_H_  ----- */
